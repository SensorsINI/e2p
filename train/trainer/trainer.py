"""
 @Time    : 29.03.22 15:59
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : trainer.py
 @Function:
 
"""
import collections
import numpy as np
import torch
# local modules
from train.base import BaseTrainer
from train.utils import inf_loop, MetricTracker
from train.utils.myutil import mean
from train.utils.training_utils import make_flow_movie, make_flow_movie_p, select_evenly_spaced_elements, make_tc_vis, make_vw_vis
from train.utils.data import data_sources
import cv2
import math

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

from train.utils.util import torch2cv2


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(len(data_loader) // 100, 1)
        self.val_log_step = max(len(valid_data_loader) // 100, 1)

        mt_keys = ['loss']
        for data_source in data_sources:
            mt_keys.append(f'loss/{data_source}')
            for l in self.loss_ftns:
                mt_keys.append(f'{l.__class__.__name__}/{data_source}')
        self.train_metrics = MetricTracker(*mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(*mt_keys, writer=self.writer)

        self.num_previews = config['trainer']['num_previews']
        self.val_num_previews = config['trainer'].get('val_num_previews', self.num_previews)
        self.val_preview_indices = select_evenly_spaced_elements(self.val_num_previews, len(self.valid_data_loader))
        self.valid_only = config['trainer'].get('valid_only', False)
        self.true_once = True  # True at init, turns False at end of _train_epoch

    def to_device(self, item):
        events = item['events'].float().to(self.device)
        image = item['frame'].float().to(self.device)
        flow = None if item['flow'] is None else item['flow'].float().to(self.device)
        return events, image, flow

    def forward_sequence(self, sequence, all_losses=False):
        losses = collections.defaultdict(list)
        self.model.reset_states()
        for i, item in enumerate(sequence):
            # print('item is', item.shape)
            # print('item events is', item['events'].shape)
            # print('item frame is', item['frame'].shape)
            # print('item flow is', item['flow'].shape)
            # print('item flow is', item['flow'])
            # exit(0)
            # events, image, flow = self.to_device(item)
            events, image, flow = self.to_device(item)
            # if torch.isnan(events).any() or torch.isinf(events).any():
            #     print('events is Nan or Inf')
            # else:
            #     print('events is no problem')
            pred = self.model(events)
            # if torch.isnan(pred['image']).any() or torch.isinf(pred['image']).any():
            #     print('pred is Nan or Inf')
            # else:
            #     print('pred is no problem')

            # i90 = raw[:, :, 0::2, 0::2]
            # i45 = raw[:, :, 0::2, 1::2]
            # i135 = raw[:, :, 1::2, 0::2]
            # i0 = raw[:, :, 1::2, 1::2]
            # s0 = i0 + i90
            # s1 = i0 - i90
            # s2 = i45 - i135

            # intensity = s0 / 2

            # aolp = 0.5 * torch.arctan2(s2, s1)
            # aolp[s2 < 0] += math.pi
            # aolp = aolp / math.pi
            # aolp = aolp + 0.5
            # image = aolp

            # dolp = torch.full_like(s0, fill_value=0)
            # mask = (s0 != 0)
            # dolp[mask] = torch.div(torch.sqrt(torch.square(s1[mask]) + torch.square(s2[mask])), s0[mask])
            # dolp = torch.clamp(dolp, min=-0, max=1)
            # image = dolp

            for loss_ftn in self.loss_ftns:
                loss_name = loss_ftn.__class__.__name__
                tmp_weight = loss_ftn.weight
                if all_losses:
                    loss_ftn.weight = 1.0
                if loss_name == 'perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred['image'], image, normalize=True))
                if loss_name == 'l2_loss':
                    losses[loss_name].append(loss_ftn(pred['image'], image))
                if loss_name == 'ssim_loss':
                    losses[loss_name].append(loss_ftn(pred['image'], image))

                if loss_name == 'temporal_consistency_loss':
                    l = loss_ftn(i, image, pred['image'], flow)
                    if l is not None:
                        losses[loss_name].append(l)
                if loss_name in ['flow_loss', 'flow_l1_loss'] and flow is not None:
                    losses[loss_name].append(loss_ftn(pred['flow'], flow))
                if loss_name == 'warping_flow_loss':
                    l = loss_ftn(i, image, pred['flow'])
                    if l is not None:
                        losses[loss_name].append(l)
                if loss_name == 'voxel_warp_flow_loss' and flow is not None:
                    losses[loss_name].append(loss_ftn(events, pred['flow']))
                if loss_name == 'flow_perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred['flow'], flow))
                if loss_name == 'combined_perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred['image'], pred['flow'], image, flow))
                loss_ftn.weight = tmp_weight
        idx = int(item['data_source_idx'].mode().values.item())
        data_source = data_sources[idx]
        losses = {f'{k}/{data_source}': mean(v) for k, v in losses.items()}
        losses['loss'] = sum(losses.values())
        losses[f'loss/{data_source}'] = losses['loss']
        return losses

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.valid_only:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                return {'val_' + k: v for k, v in val_log.items()}
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence)
            loss = losses['loss']
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for k, v in losses.items():
                self.train_metrics.update(k, v.item())

            if batch_idx % self.log_step == 0:
                msg = 'Train Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx < self.num_previews and (epoch - 1) % self.save_period == 0:
                with torch.no_grad():
                    self.preview(sequence, epoch, tag_prefix=f'train_{batch_idx}')

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        print("validation")
        if self.do_validation and epoch % 10 == 0:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.true_once = False
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        i = 0
        for batch_idx, sequence in enumerate(self.valid_data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence, all_losses=True)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            for k, v in losses.items():
                self.valid_metrics.update(k, v.item())

            if batch_idx % self.val_log_step == 0:
                msg = 'Valid Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.valid_data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx in self.val_preview_indices and (epoch - 1) % self.save_period == 0:
                self.preview(sequence, epoch, tag_prefix=f'val_{i}')
                i += 1

        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)

    def preview(self, sequence, epoch, tag_prefix=''):
        """
        Plot visualisation to tensorboard.
        Plots input, output, groundtruth histograms and movies
        """
        print(f'Making preview {tag_prefix}')
        event_previews, pred_flows, pred_images, flows, images, voxels = [], [], [], [], [], []
        self.model.reset_states()
        for i, item in enumerate(sequence):
            item = {k: v[0:1, ...] for k, v in item.items()}  # set batch size to 1
            events, image, flow = self.to_device(item)
            pred = self.model(events)
            event_previews.append(torch.sum(events, dim=1, keepdim=True))
            pred_flows.append(pred.get('flow', 0 * flow))
            pred_images.append(pred['image'])
            flows.append(flow)
            images.append(pred['image'])
            voxels.append(events)

        tc_loss_ftn = self.get_loss_ftn('temporal_consistency_loss')
        if self.true_once and tc_loss_ftn is not None:
            for i, image in enumerate(images):
                output = tc_loss_ftn(i, image, pred_images[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_vis/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break

        vw_loss_ftn = self.get_loss_ftn('voxel_warp_flow_loss')
        if self.true_once and vw_loss_ftn is not None:
            for i, image in enumerate(images):
                output = vw_loss_ftn(voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_vox/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break

        non_zero_voxel = torch.stack([s['events'] for s in sequence])
        non_zero_voxel = non_zero_voxel[non_zero_voxel != 0]
        if torch.numel(non_zero_voxel) == 0:
            non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_flow/groundtruth',
                                  torch.stack(flows))
        self.writer.add_histogram(f'{tag_prefix}_image/groundtruth',
                                  torch.stack(images))
        self.writer.add_histogram(f'{tag_prefix}_input',
                                  non_zero_voxel)
        self.writer.add_histogram(f'{tag_prefix}_flow/prediction',
                                  torch.stack(pred_flows))
        self.writer.add_histogram(f'{tag_prefix}_image/prediction',
                                  torch.stack(pred_images))
        video_tensor = make_flow_movie(event_previews, pred_images, images, pred_flows, flows)
        self.writer.writer.add_video(f'{tag_prefix}', video_tensor, global_step=epoch, fps=20)

    def get_loss_ftn(self, loss_name):
        for loss_ftn in self.loss_ftns:
            if loss_ftn.__class__.__name__ == loss_name:
                return loss_ftn
        return None


class Trainer_I(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(len(data_loader) // 100, 1)
        self.val_log_step = max(len(valid_data_loader) // 100, 1)

        mt_keys = ['loss']
        for data_source in data_sources:
            mt_keys.append(f'loss/{data_source}')
            for l in self.loss_ftns:
                mt_keys.append(f'{l.__class__.__name__}/{data_source}')
                for p in ['i90', 'i45', 'i135', 'i0']:
                # for p in ['s0', 's1', 's2', 'i', 'a', 'd']:
                    mt_keys.append(f'{p}/{l.__class__.__name__}/{data_source}')
        self.train_metrics = MetricTracker(*mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(*mt_keys, writer=self.writer)

        self.num_previews = config['trainer']['num_previews']
        self.val_num_previews = config['trainer'].get('val_num_previews', self.num_previews)
        self.val_preview_indices = select_evenly_spaced_elements(self.val_num_previews, len(self.valid_data_loader))
        self.valid_only = config['trainer'].get('valid_only', False)
        self.true_once = True  # True at init, turns False at end of _train_epoch

    def to_device(self, item):
        events = item['events'].float().to(self.device)
        # s0 = item['s0'].float().to(self.device)
        # s1 = item['s1'].float().to(self.device)
        # s2 = item['s2'].float().to(self.device)
        # i = item['intensity'].float().to(self.device)
        # a = item['aolp'].float().to(self.device)
        # d = item['dolp'].float().to(self.device)
        flow = None if item['flow'] is None else item['flow'].float().to(self.device)
        # return events, s0, s1, s2, i, a, d, flow
        raw = item['raw'].float().to(self.device)
        return events, raw, flow

    def forward_sequence(self, sequence, all_losses=False):
        losses = collections.defaultdict(list)
        # self.model.reset_states_s0()
        # self.model.reset_states_s1()
        # self.model.reset_states_s2()
        self.model.reset_states_i90()
        self.model.reset_states_i45()
        self.model.reset_states_i135()
        self.model.reset_states_i0()
        for i, item in enumerate(sequence):
            # events, s0, s1, s2, i, a, d, flow = self.to_device(item)
            events, raw, flow = self.to_device(item)

            i90 = raw[:, :, 0::2, 0::2]
            i45 = raw[:, :, 0::2, 1::2]
            i135 = raw[:, :, 1::2, 0::2]
            i0 = raw[:, :, 1::2, 1::2]

            pred = self.model(events)

            for loss_ftn in self.loss_ftns:
                loss_name = loss_ftn.__class__.__name__
                tmp_weight = loss_ftn.weight
                if all_losses:
                    loss_ftn.weight = 1.0
                if loss_name == 'perceptual_loss':
                    losses[f'i90/{loss_name}'].append(loss_ftn(pred['i90'], i90, normalize=True))
                    losses[f'i45/{loss_name}'].append(loss_ftn(pred['i45'], i45, normalize=True))
                    losses[f'i135/{loss_name}'].append(loss_ftn(pred['i135'], i135, normalize=True))
                    losses[f'i0/{loss_name}'].append(loss_ftn(pred['i0'], i0, normalize=True))
                #     losses[f's1/{loss_name}'].append(loss_ftn(pred['s1'], s1, normalize=True))
                #     losses[f's2/{loss_name}'].append(loss_ftn(pred['s2'], s2, normalize=True))
                # if loss_name == 'mse_loss':
                #     losses[f's0/{loss_name}'].append(loss_ftn(pred['s0'], s0))
                #     losses[f's1/{loss_name}'].append(loss_ftn(pred['s1'], s1) * 10)
                #     losses[f's2/{loss_name}'].append(loss_ftn(pred['s2'], s2) * 10)
                #     losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], a))
                #     losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], d))
                # if loss_name == 'ms_ssim_loss':
                #     losses[f's0/{loss_name}'].append(loss_ftn(pred['s0'], s0))
                #     losses[f's1/{loss_name}'].append(loss_ftn(pred['s1'], s1) * 10)
                #     losses[f's2/{loss_name}'].append(loss_ftn(pred['s2'], s2) * 10)
                # if loss_name == 'temporal_consistency_loss':
                #     l = loss_ftn(i, image, pred['image'], flow)
                #     if l is not None:
                #         losses[loss_name].append(l)
                # if loss_name in ['flow_loss', 'flow_l1_loss'] and flow is not None:
                #     losses[loss_name].append(loss_ftn(pred['flow'], flow))
                # if loss_name == 'warping_flow_loss':
                #     l = loss_ftn(i, image, pred['flow'])
                #     if l is not None:
                #         losses[loss_name].append(l)
                # if loss_name == 'voxel_warp_flow_loss' and flow is not None:
                #     losses[loss_name].append(loss_ftn(events, pred['flow']))
                # if loss_name == 'flow_perceptual_loss':
                #     losses[loss_name].append(loss_ftn(pred['flow'], flow))
                # if loss_name == 'combined_perceptual_loss':
                #     losses[loss_name].append(loss_ftn(pred['image'], pred['flow'], image, flow))
                loss_ftn.weight = tmp_weight
        idx = int(item['data_source_idx'].mode().values.item())
        data_source = data_sources[idx]
        losses = {f'{k}/{data_source}': mean(v) for k, v in losses.items()}
        losses['loss'] = sum(losses.values())
        losses[f'loss/{data_source}'] = losses['loss']
        return losses

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.valid_only:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                return {'val_' + k: v for k, v in val_log.items()}
        self.model.train()
        self.train_metrics.reset()
        # scaler = GradScaler()
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence)
            loss = losses['loss']
            # with autocast():
            #     losses = self.forward_sequence(sequence)
            #     loss = losses['loss']
            # original
            loss.backward()
            self.optimizer.step()
            # for autocast
            # scaler.scale(loss).backward()
            # scaler.step(self.optimizer)
            # scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for k, v in losses.items():
                self.train_metrics.update(k, v.item())

            if batch_idx % self.log_step == 0:
                msg = 'Train Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx < self.num_previews and (epoch - 1) % self.save_period == 0:
                with torch.no_grad():
                    self.preview(sequence, epoch, tag_prefix=f'train_{batch_idx}')

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        print("validation")
        if self.do_validation and epoch % 10 == 0:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.true_once = False
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        i = 0
        for batch_idx, sequence in enumerate(self.valid_data_loader):
            self.optimizer.zero_grad()
            # original
            losses = self.forward_sequence(sequence, all_losses=True)
            # autocast
            # with autocast():
            #     losses = self.forward_sequence(sequence, all_losses=True)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            for k, v in losses.items():
                self.valid_metrics.update(k, v.item())

            if batch_idx % self.val_log_step == 0:
                msg = 'Valid Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.valid_data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx in self.val_preview_indices and (epoch - 1) % self.save_period == 0:
                self.preview(sequence, epoch, tag_prefix=f'val_{i}')
                i += 1

        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)

    def preview(self, sequence, epoch, tag_prefix=''):
        """
        Plot visualisation to tensorboard.
        Plots input, output, groundtruth histograms and movies
        """
        print(f'Making preview {tag_prefix}')
        positive_event_previews, negative_event_previews, positive_voxels, negative_voxels, \
        pred_flows, pred_intensities, pred_aolps, pred_dolps, \
        flows, intensities, aolps, dolps = [], [], [], [], [], [], [], [], [], [], [], []
        # self.model.reset_states_i()
        self.model.reset_states_i90()
        self.model.reset_states_i45()
        self.model.reset_states_i135()
        self.model.reset_states_i0()
        # self.model.reset_states_i()
        # self.model.reset_states_a()
        # self.model.reset_states_d()
        # self.model.reset_states_s0()
        # self.model.reset_states_s1()
        # self.model.reset_states_s2()
        # self.model.reset_states_intensity()
        # self.model.reset_states_aolp()
        # self.model.reset_states_dolp()
        for i, item in enumerate(sequence):
            item = {k: v[0:1, ...] for k, v in item.items()}  # set batch size to 1
            # events, intensity, aolp, dolp, flow = self.to_device(item)
            # events, s0, s1, s2, i, a, d, flow = self.to_device(item)
            events, raw, flow = self.to_device(item)
            # events, image, flow = self.to_device(item)

            # intensity = s0
            # aolp = s1
            # dolp = s2
            intensity = raw[:, :, 0::2, 0::2]
            aolp = raw[:, :, 0::2, 0::2]
            dolp = raw[:, :, 0::2, 0::2]

            pred = self.model(events)

            positive_event_previews.append(torch.sum(events[:, 0:events.shape[1] // 2, :, :], dim=1, keepdim=True))
            negative_event_previews.append(torch.sum(events[:, events.shape[1] // 2:-1, :, :], dim=1, keepdim=True))
            positive_voxels.append(events[:, 0:events.shape[1] // 2, :, :])
            negative_voxels.append(events[:, events.shape[1] // 2:-1, :, :])

            pred_flows.append(pred.get('flow', 0 * flow))
            # pred_intensities.append(pred['i_90'])
            # pred_aolps.append(pred['i_45'])
            # pred_dolps.append(pred['i_135'])
            # pred_intensities.append(pred['s0'])
            # pred_aolps.append(pred['s1'])
            # pred_dolps.append(pred['s2'])
            pred_intensities.append(pred['i90'])
            pred_aolps.append(pred['i45'])
            pred_dolps.append(pred['i135'])

            flows.append(flow)
            intensities.append(intensity)
            aolps.append(aolp)
            dolps.append(dolp)

        tc_loss_ftn = self.get_loss_ftn('temporal_consistency_loss')
        if self.true_once and tc_loss_ftn is not None:
            for i, intensity in enumerate(intensities):
                output = tc_loss_ftn(i, intensity, pred_intensities[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_intensity/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break
            for i, aolp in enumerate(aolps):
                output = tc_loss_ftn(i, aolp, pred_aolps[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_aolp/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break
            for i, dolp in enumerate(dolps):
                output = tc_loss_ftn(i, dolp, pred_dolps[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_dolp/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break

        vw_loss_ftn = self.get_loss_ftn('voxel_warp_flow_loss')
        if self.true_once and vw_loss_ftn is not None:
            for i, intensity in enumerate(intensities):
                output = vw_loss_ftn(positive_voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_positive_voxels/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break
            for i, intensity in enumerate(intensities):
                output = vw_loss_ftn(negative_voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_negative_voxels/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break

        # histogram
        self.writer.add_histogram(f'{tag_prefix}_flow/groundtruth', torch.stack(flows))
        self.writer.add_histogram(f'{tag_prefix}_intensity/groundtruth', torch.stack(intensities))
        self.writer.add_histogram(f'{tag_prefix}_aolp/groundtruth', torch.stack(aolps))
        self.writer.add_histogram(f'{tag_prefix}_dolp/groundtruth', torch.stack(dolps))

        positive_non_zero_voxel = torch.stack([s['events'][:, 0:s['events'].shape[1] // 2, :, :] for s in sequence])
        positive_non_zero_voxel = positive_non_zero_voxel[positive_non_zero_voxel != 0]
        if torch.numel(positive_non_zero_voxel) == 0:
            positive_non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_input/positive', positive_non_zero_voxel)

        negative_non_zero_voxel = torch.stack([s['events'][:, s['events'].shape[1] // 2:-1, :, :] for s in sequence])
        negative_non_zero_voxel = negative_non_zero_voxel[negative_non_zero_voxel != 0]
        if torch.numel(negative_non_zero_voxel) == 0:
            negative_non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_input/negative', negative_non_zero_voxel)

        self.writer.add_histogram(f'{tag_prefix}_flow/prediction', torch.stack(pred_flows))
        self.writer.add_histogram(f'{tag_prefix}_intensity/prediction', torch.stack(pred_intensities))
        self.writer.add_histogram(f'{tag_prefix}_aolp/prediction', torch.stack(pred_aolps))
        self.writer.add_histogram(f'{tag_prefix}_dolp/prediction', torch.stack(pred_dolps))

        video_tensor = make_flow_movie_p(positive_event_previews, negative_event_previews,
                                         pred_intensities, intensities, pred_aolps, aolps, pred_dolps, dolps,
                                         pred_flows, flows)
        self.writer.writer.add_video(f'{tag_prefix}', video_tensor, global_step=epoch, fps=20)

    def get_loss_ftn(self, loss_name):
        for loss_ftn in self.loss_ftns:
            if loss_ftn.__class__.__name__ == loss_name:
                return loss_ftn
        return None


class Trainer_S(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(len(data_loader) // 100, 1)
        self.val_log_step = max(len(valid_data_loader) // 100, 1)

        mt_keys = ['loss']
        for data_source in data_sources:
            mt_keys.append(f'loss/{data_source}')
            for l in self.loss_ftns:
                mt_keys.append(f'{l.__class__.__name__}/{data_source}')
                for p in ['s0', 's1', 's2']:
                # for p in ['s0', 's1', 's2', 'i', 'a', 'd']:
                    mt_keys.append(f'{p}/{l.__class__.__name__}/{data_source}')
        self.train_metrics = MetricTracker(*mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(*mt_keys, writer=self.writer)

        self.num_previews = config['trainer']['num_previews']
        self.val_num_previews = config['trainer'].get('val_num_previews', self.num_previews)
        self.val_preview_indices = select_evenly_spaced_elements(self.val_num_previews, len(self.valid_data_loader))
        self.valid_only = config['trainer'].get('valid_only', False)
        self.true_once = True  # True at init, turns False at end of _train_epoch

    def to_device(self, item):
        events = item['events'].float().to(self.device)
        # s0 = item['s0'].float().to(self.device)
        # s1 = item['s1'].float().to(self.device)
        # s2 = item['s2'].float().to(self.device)
        # i = item['intensity'].float().to(self.device)
        # a = item['aolp'].float().to(self.device)
        # d = item['dolp'].float().to(self.device)
        flow = None if item['flow'] is None else item['flow'].float().to(self.device)
        # return events, s0, s1, s2, i, a, d, flow
        raw = item['raw'].float().to(self.device)
        return events, raw, flow

    def forward_sequence(self, sequence, all_losses=False):
        losses = collections.defaultdict(list)
        # self.model.reset_states_s0()
        # self.model.reset_states_s1()
        # self.model.reset_states_s2()
        self.model.reset_states_i()
        self.model.reset_states_a()
        self.model.reset_states_d()
        for i, item in enumerate(sequence):
            # events, s0, s1, s2, i, a, d, flow = self.to_device(item)
            events, raw, flow = self.to_device(item)

            i90 = raw[:, :, 0::2, 0::2]
            i45 = raw[:, :, 0::2, 1::2]
            i135 = raw[:, :, 1::2, 0::2]
            i0 = raw[:, :, 1::2, 1::2]
            s0 = i0 + i90
            # remember to double the output when test
            s0 = s0 / 2
            s1 = i0 - i90
            s2 = i45 - i135

            pred = self.model(events)

            for loss_ftn in self.loss_ftns:
                loss_name = loss_ftn.__class__.__name__
                tmp_weight = loss_ftn.weight
                if all_losses:
                    loss_ftn.weight = 1.0
                if loss_name == 'perceptual_loss':
                    losses[f's0/{loss_name}'].append(loss_ftn(pred['i'], s0, normalize=True))
                    losses[f's1/{loss_name}'].append(loss_ftn(pred['a'], s1, normalize=False))
                    losses[f's2/{loss_name}'].append(loss_ftn(pred['d'], s2, normalize=False))
                #     losses[f's1/{loss_name}'].append(loss_ftn(pred['s1'], s1, normalize=True))
                #     losses[f's2/{loss_name}'].append(loss_ftn(pred['s2'], s2, normalize=True))
                # if loss_name == 'mse_loss':
                #     losses[f's0/{loss_name}'].append(loss_ftn(pred['s0'], s0))
                #     losses[f's1/{loss_name}'].append(loss_ftn(pred['s1'], s1) * 10)
                #     losses[f's2/{loss_name}'].append(loss_ftn(pred['s2'], s2) * 10)
                #     losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], a))
                #     losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], d))
                # if loss_name == 'ms_ssim_loss':
                #     losses[f's0/{loss_name}'].append(loss_ftn(pred['s0'], s0))
                #     losses[f's1/{loss_name}'].append(loss_ftn(pred['s1'], s1) * 10)
                #     losses[f's2/{loss_name}'].append(loss_ftn(pred['s2'], s2) * 10)
                # if loss_name == 'temporal_consistency_loss':
                #     l = loss_ftn(i, image, pred['image'], flow)
                #     if l is not None:
                #         losses[loss_name].append(l)
                # if loss_name in ['flow_loss', 'flow_l1_loss'] and flow is not None:
                #     losses[loss_name].append(loss_ftn(pred['flow'], flow))
                # if loss_name == 'warping_flow_loss':
                #     l = loss_ftn(i, image, pred['flow'])
                #     if l is not None:
                #         losses[loss_name].append(l)
                # if loss_name == 'voxel_warp_flow_loss' and flow is not None:
                #     losses[loss_name].append(loss_ftn(events, pred['flow']))
                # if loss_name == 'flow_perceptual_loss':
                #     losses[loss_name].append(loss_ftn(pred['flow'], flow))
                # if loss_name == 'combined_perceptual_loss':
                #     losses[loss_name].append(loss_ftn(pred['image'], pred['flow'], image, flow))
                loss_ftn.weight = tmp_weight
        idx = int(item['data_source_idx'].mode().values.item())
        data_source = data_sources[idx]
        losses = {f'{k}/{data_source}': mean(v) for k, v in losses.items()}
        losses['loss'] = sum(losses.values())
        losses[f'loss/{data_source}'] = losses['loss']
        return losses

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.valid_only:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                return {'val_' + k: v for k, v in val_log.items()}
        self.model.train()
        self.train_metrics.reset()
        # scaler = GradScaler()
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence)
            loss = losses['loss']
            # with autocast():
            #     losses = self.forward_sequence(sequence)
            #     loss = losses['loss']
            # original
            loss.backward()
            self.optimizer.step()
            # for autocast
            # scaler.scale(loss).backward()
            # scaler.step(self.optimizer)
            # scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for k, v in losses.items():
                self.train_metrics.update(k, v.item())

            if batch_idx % self.log_step == 0:
                msg = 'Train Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx < self.num_previews and (epoch - 1) % self.save_period == 0:
                with torch.no_grad():
                    self.preview(sequence, epoch, tag_prefix=f'train_{batch_idx}')

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        print("validation")
        if self.do_validation and epoch % 10 == 0:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.true_once = False
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        i = 0
        for batch_idx, sequence in enumerate(self.valid_data_loader):
            self.optimizer.zero_grad()
            # original
            losses = self.forward_sequence(sequence, all_losses=True)
            # autocast
            # with autocast():
            #     losses = self.forward_sequence(sequence, all_losses=True)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            for k, v in losses.items():
                self.valid_metrics.update(k, v.item())

            if batch_idx % self.val_log_step == 0:
                msg = 'Valid Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.valid_data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx in self.val_preview_indices and (epoch - 1) % self.save_period == 0:
                self.preview(sequence, epoch, tag_prefix=f'val_{i}')
                i += 1

        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)

    def preview(self, sequence, epoch, tag_prefix=''):
        """
        Plot visualisation to tensorboard.
        Plots input, output, groundtruth histograms and movies
        """
        print(f'Making preview {tag_prefix}')
        positive_event_previews, negative_event_previews, positive_voxels, negative_voxels, \
        pred_flows, pred_intensities, pred_aolps, pred_dolps, \
        flows, intensities, aolps, dolps = [], [], [], [], [], [], [], [], [], [], [], []
        # self.model.reset_states_i()
        # self.model.reset_states_90()
        # self.model.reset_states_45()
        # self.model.reset_states_135()
        # self.model.reset_states_0()
        self.model.reset_states_i()
        self.model.reset_states_a()
        self.model.reset_states_d()
        # self.model.reset_states_s0()
        # self.model.reset_states_s1()
        # self.model.reset_states_s2()
        # self.model.reset_states_intensity()
        # self.model.reset_states_aolp()
        # self.model.reset_states_dolp()
        for i, item in enumerate(sequence):
            item = {k: v[0:1, ...] for k, v in item.items()}  # set batch size to 1
            # events, intensity, aolp, dolp, flow = self.to_device(item)
            # events, s0, s1, s2, i, a, d, flow = self.to_device(item)
            events, raw, flow = self.to_device(item)
            # events, image, flow = self.to_device(item)

            # intensity = s0
            # aolp = s1
            # dolp = s2
            intensity = raw[:, :, 0::2, 0::2]
            aolp = raw[:, :, 0::2, 0::2]
            dolp = raw[:, :, 0::2, 0::2]

            pred = self.model(events)

            positive_event_previews.append(torch.sum(events[:, 0:events.shape[1] // 2, :, :], dim=1, keepdim=True))
            negative_event_previews.append(torch.sum(events[:, events.shape[1] // 2:-1, :, :], dim=1, keepdim=True))
            positive_voxels.append(events[:, 0:events.shape[1] // 2, :, :])
            negative_voxels.append(events[:, events.shape[1] // 2:-1, :, :])

            pred_flows.append(pred.get('flow', 0 * flow))
            # pred_intensities.append(pred['i_90'])
            # pred_aolps.append(pred['i_45'])
            # pred_dolps.append(pred['i_135'])
            # pred_intensities.append(pred['s0'])
            # pred_aolps.append(pred['s1'])
            # pred_dolps.append(pred['s2'])
            pred_intensities.append(pred['i'])
            pred_aolps.append(pred['a'])
            pred_dolps.append(pred['d'])

            flows.append(flow)
            intensities.append(intensity)
            aolps.append(aolp)
            dolps.append(dolp)

        tc_loss_ftn = self.get_loss_ftn('temporal_consistency_loss')
        if self.true_once and tc_loss_ftn is not None:
            for i, intensity in enumerate(intensities):
                output = tc_loss_ftn(i, intensity, pred_intensities[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_intensity/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break
            for i, aolp in enumerate(aolps):
                output = tc_loss_ftn(i, aolp, pred_aolps[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_aolp/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break
            for i, dolp in enumerate(dolps):
                output = tc_loss_ftn(i, dolp, pred_dolps[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_dolp/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break

        vw_loss_ftn = self.get_loss_ftn('voxel_warp_flow_loss')
        if self.true_once and vw_loss_ftn is not None:
            for i, intensity in enumerate(intensities):
                output = vw_loss_ftn(positive_voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_positive_voxels/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break
            for i, intensity in enumerate(intensities):
                output = vw_loss_ftn(negative_voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_negative_voxels/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break

        # histogram
        self.writer.add_histogram(f'{tag_prefix}_flow/groundtruth', torch.stack(flows))
        self.writer.add_histogram(f'{tag_prefix}_intensity/groundtruth', torch.stack(intensities))
        self.writer.add_histogram(f'{tag_prefix}_aolp/groundtruth', torch.stack(aolps))
        self.writer.add_histogram(f'{tag_prefix}_dolp/groundtruth', torch.stack(dolps))

        positive_non_zero_voxel = torch.stack([s['events'][:, 0:s['events'].shape[1] // 2, :, :] for s in sequence])
        positive_non_zero_voxel = positive_non_zero_voxel[positive_non_zero_voxel != 0]
        if torch.numel(positive_non_zero_voxel) == 0:
            positive_non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_input/positive', positive_non_zero_voxel)

        negative_non_zero_voxel = torch.stack([s['events'][:, s['events'].shape[1] // 2:-1, :, :] for s in sequence])
        negative_non_zero_voxel = negative_non_zero_voxel[negative_non_zero_voxel != 0]
        if torch.numel(negative_non_zero_voxel) == 0:
            negative_non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_input/negative', negative_non_zero_voxel)

        self.writer.add_histogram(f'{tag_prefix}_flow/prediction', torch.stack(pred_flows))
        self.writer.add_histogram(f'{tag_prefix}_intensity/prediction', torch.stack(pred_intensities))
        self.writer.add_histogram(f'{tag_prefix}_aolp/prediction', torch.stack(pred_aolps))
        self.writer.add_histogram(f'{tag_prefix}_dolp/prediction', torch.stack(pred_dolps))

        video_tensor = make_flow_movie_p(positive_event_previews, negative_event_previews,
                                         pred_intensities, intensities, pred_aolps, aolps, pred_dolps, dolps,
                                         pred_flows, flows)
        self.writer.writer.add_video(f'{tag_prefix}', video_tensor, global_step=epoch, fps=20)

    def get_loss_ftn(self, loss_name):
        for loss_ftn in self.loss_ftns:
            if loss_ftn.__class__.__name__ == loss_name:
                return loss_ftn
        return None


class Trainer_P(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(len(data_loader) // 100, 1)
        self.val_log_step = max(len(valid_data_loader) // 100, 1)

        mt_keys = ['loss']
        for data_source in data_sources:
            mt_keys.append(f'loss/{data_source}')
            for l in self.loss_ftns:
                mt_keys.append(f'{l.__class__.__name__}/{data_source}')
                for p in ['i', 'a', 'd']:
                    mt_keys.append(f'{p}/{l.__class__.__name__}/{data_source}')
        self.train_metrics = MetricTracker(*mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(*mt_keys, writer=self.writer)

        self.num_previews = config['trainer']['num_previews']
        self.val_num_previews = config['trainer'].get('val_num_previews', self.num_previews)
        self.val_preview_indices = select_evenly_spaced_elements(self.val_num_previews, len(self.valid_data_loader))
        self.valid_only = config['trainer'].get('valid_only', False)
        self.true_once = True  # True at init, turns False at end of _train_epoch

    def to_device(self, item):
        events = item['events'].float().to(self.device)
        intensity = item['intensity'].float().to(self.device)
        aolp = item['aolp'].float().to(self.device)
        dolp = item['dolp'].float().to(self.device)
        flow = None if item['flow'] is None else item['flow'].float().to(self.device)

        return events, intensity, aolp, dolp, flow

    def forward_sequence(self, sequence, all_losses=False):
        losses = collections.defaultdict(list)
        # self.model.reset_states()
        # self.model.reset_states_90()
        # self.model.reset_states_45()
        # self.model.reset_states_135()
        # self.model.reset_states_0()
        # self.model.reset_states_s0()
        # self.model.reset_states_s1()
        # self.model.reset_states_s2()
        self.model.reset_states_i()
        self.model.reset_states_a()
        self.model.reset_states_d()
        # self.model.firenet_i90.reset_states()
        # self.model.firenet_i45.reset_states()
        # self.model.firenet_i135.reset_states()
        # self.model.firenet_i0.reset_states()
        # self.model.reset_states_i_shared()
        for i, item in enumerate(sequence):
            # get polarization from raw computation
            # events, image, flow = self.to_device(item)
            #
            # pred = self.model(events)
            #
            # i90 = image[:, :, 0::2, 0::2]
            # i45 = image[:, :, 0::2, 1::2]
            # i135 = image[:, :, 1::2, 0::2]
            # i0 = image[:, :, 1::2, 1::2]
            # s0 = i0 + i90
            # s1 = i0 - i90
            # s2 = i45 - i135
            #
            # intensity = s0 / 2
            #
            # aolp = 0.5 * torch.arctan2(s2, s1)
            # aolp = aolp + 0.5 * math.pi
            # aolp = aolp / math.pi
            #
            # dolp = torch.full_like(s0, fill_value=0)
            # mask = (s0 != 0)
            # dolp[mask] = torch.div(torch.sqrt(torch.square(s1[mask]) + torch.square(s2[mask])), s0[mask])
            # dolp = torch.clamp(dolp, min=-0, max=1)
            #
            # aolpdolp = aolp * dolp

            # get polarization from dataloader directly
            events, intensity, aolp, dolp, flow = self.to_device(item)
            # print(f"aolp min:{aolp.min()} aolp max:{aolp.max()}")
            # print(f'events shape:{events.shape} max:{events.max()} min:{events.min()}')
            # print('----')
            # f0 = events[0, 0, :, :].cpu().numpy()
            # print(np.max(f0), np.min(f0))
            # f0 = (f0 - np.min(f0)) / (np.max(f0) - np.min(f0))
            # f1 = events[0, 1, :, :].cpu().numpy()
            # print(np.max(f1), np.min(f1))
            # f1 = (f1 - np.min(f1)) / (np.max(f1) - np.min(f1))
            # f2 = events[0, 2, :, :].cpu().numpy()
            # print(np.max(f2), np.min(f2))
            # f2 = (f2 - np.min(f2)) / (np.max(f2) - np.min(f2))
            # f3 = events[0, 3, :, :].cpu().numpy()
            # print(np.max(f3), np.min(f3))
            # f3 = (f3 - np.min(f3)) / (np.max(f3) - np.min(f3))
            # f4 = events[0, 4, :, :].cpu().numpy()
            # print(np.max(f4), np.min(f4))
            # f4 = (f4 - np.min(f4)) / (np.max(f4) - np.min(f4))
            #
            # f = cv2.hconcat([f0, f1, f2, f3, f4])
            # cv2.imshow('f', f)
            # cv2.waitKey(0)

            pred = self.model(events)

            # aolp_w = aolp[:, :, :, :-1] - aolp[:, :, :, 1:]
            # aolp_h = aolp[:, :, :-1, :] - aolp[:, :, 1:, :]
            # pred_aolp_w = pred['a'][:, :, :, :-1] - pred['a'][:, :, :, 1:]
            # pred_aolp_h = pred['a'][:, :, :-1, :] - pred['a'][:, :, 1:, :]

            for loss_ftn in self.loss_ftns:
                loss_name = loss_ftn.__class__.__name__
                tmp_weight = loss_ftn.weight
                if all_losses:
                    loss_ftn.weight = 1.0
                if loss_name == 'perceptual_loss':
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity, normalize=True))
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity, normalize=True) * 2)
                    losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity, normalize=True) * 3)
                    losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp, normalize=True))
                    losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp, normalize=True))
                if loss_name == 'mse_loss':
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                    # losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                    losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                if loss_name == 'mse_loss_aolp':
                    losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                if loss_name == 'abs_sin_loss':
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                    losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                    # losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                if loss_name == 'ssim_loss':
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                    losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                    losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                # if loss_name == 'l2_dw_loss':
                #     losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp, dolp, 'aolp'))
                #     losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], aolp, dolp, 'dolp'))
                    # circular angle loss
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                    # losses[f'ac/{loss_name}'].append(loss_ftn((torch.cos(2 * math.pi * pred['a']) + 1) / 2, (torch.cos(2 * math.pi * aolp) + 1) / 2))
                    # losses[f'as/{loss_name}'].append(loss_ftn((torch.sin(2 * math.pi * pred['a']) + 1) / 2, (torch.sin(2 * math.pi * aolp) + 1) / 2))
                    # losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                    # add reverse loss
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], 1 - intensity))
                    # losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], 1 - aolp))
                    # losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], 1 - dolp))
                    # losses[f'i_s/{loss_name}'].append(loss_ftn(pred['i_s'], intensity))
                    # losses[f'i_f/{loss_name}'].append(loss_ftn(pred['i_f'], intensity))
                    # losses[f'a_s/{loss_name}'].append(loss_ftn(pred['a_s'], aolp))
                    # losses[f'a_f/{loss_name}'].append(loss_ftn(pred['a_f'], aolp))
                    # losses[f'd_s/{loss_name}'].append(loss_ftn(pred['d_s'], dolp))
                    # losses[f'd_f/{loss_name}'].append(loss_ftn(pred['d_f'], dolp))
                    # losses[f'aw/{loss_name}'].append(loss_ftn(pred_aolp_w, aolp_w))
                    # losses[f'ah/{loss_name}'].append(loss_ftn(pred_aolp_h, aolp_h))
                # if loss_name == 'ms_ssim_loss':
                    # losses[f'90/{loss_name}'].append(loss_ftn(pred['i_90'], i90))
                    # losses[f'45/{loss_name}'].append(loss_ftn(pred['i_45'], i45))
                    # losses[f'135/{loss_name}'].append(loss_ftn(pred['i_135'], i135))
                    # losses[f'0/{loss_name}'].append(loss_ftn(pred['i_0'], i0))
                    # normal
                    # losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                    # losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                # if loss_name == 'pae_loss':
                #     losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                #     losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                    # circular angle loss
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                    # losses[f'ac/{loss_name}'].append(loss_ftn((torch.cos(2 * math.pi * pred['a']) + 1) / 2, (torch.cos(2 * math.pi * aolp) + 1) / 2))
                    # losses[f'as/{loss_name}'].append(loss_ftn((torch.sin(2 * math.pi * pred['a']) + 1) / 2, (torch.sin(2 * math.pi * aolp) + 1) / 2))
                    # losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                    # add reverse loss
                    # losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], 1 - intensity))
                    # losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], 1 - aolp))
                    # losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], 1 - dolp))
                    # losses[f'i_s/{loss_name}'].append(loss_ftn(pred['i_s'], intensity))
                    # losses[f'i_f/{loss_name}'].append(loss_ftn(pred['i_f'], intensity))
                    # losses[f'a_s/{loss_name}'].append(loss_ftn(pred['a_s'], aolp))
                    # losses[f'a_f/{loss_name}'].append(loss_ftn(pred['a_f'], aolp))
                    # losses[f'd_s/{loss_name}'].append(loss_ftn(pred['d_s'], dolp))
                    # losses[f'd_f/{loss_name}'].append(loss_ftn(pred['d_f'], dolp))
                    # losses[f'aw/{loss_name}'].append(loss_ftn(pred_aolp_w, aolp_w))
                    # losses[f'ah/{loss_name}'].append(loss_ftn(pred_aolp_h, aolp_h))
                # if loss_name == 'pae_loss':
                #     losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp, path='./ckpt_pae/aolp_100.pth'))
                #     losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp, path='./ckpt_pae/dolp_100.pth'))
                # if loss_name == 'mdf_i_loss':
                #     losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                # if loss_name == 'mdf_a_loss':
                #     losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                # if loss_name == 'mdf_d_loss':
                #     losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                # if loss_name == 'dct_loss':
                #     losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                #     losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                #     losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                # if loss_name == 'hfd_loss':
                #     losses[f'i/{loss_name}'].append(loss_ftn(pred['hfd_features_i'], pred['i'], intensity))
                #     losses[f'a/{loss_name}'].append(loss_ftn(pred['hfd_features_a'], pred['a'], aolp))
                #     losses[f'd/{loss_name}'].append(loss_ftn(pred['hfd_features_d'], pred['d'], dolp))
                # if loss_name == 'cos_loss':
                #     losses[f'aolp/{loss_name}'].append(loss_ftn(pred['aolp'], aolp))
                # if loss_name == 'temporal_consistency_loss':
                #     if math.isnan(flow.cpu().numpy().any()):
                #         print('Flow is Nan.')
                    # print('input flow')
                    # print(torch.max(flow))
                    # print(torch.min(flow))
                    # l_intensity = loss_ftn(i, intensity, pred['intensity'], flow)
                    # l_aolp = loss_ftn(i, aolp, pred['aolp'], flow)
                    # l_dolp = loss_ftn(i, dolp, pred['dolp'], flow)
                    # if l_intensity is not None:
                    #     losses[f'intensity/{loss_name}'].append(l_intensity)
                    # if l_aolp is not None:
                    #     losses[f'aolp/{loss_name}'].append(l_aolp)
                    # if l_dolp is not None:
                    #     losses[f'dolp/{loss_name}'].append(l_dolp)
                # if loss_name in ['flow_loss', 'flow_l1_loss'] and flow is not None:
                #     losses[loss_name].append(loss_ftn(pred['flow'], flow))
                # if loss_name == 'warping_flow_loss':
                #     l = loss_ftn(i, image, pred['flow'])
                #     if l is not None:
                #         losses[loss_name].append(l)
                # if loss_name == 'voxel_warp_flow_loss' and flow is not None:
                #     losses[loss_name].append(loss_ftn(events, pred['flow']))
                # if loss_name == 'flow_perceptual_loss':
                #     losses[loss_name].append(loss_ftn(pred['flow'], flow))
                # if loss_name == 'combined_perceptual_loss':
                #     losses[loss_name].append(loss_ftn(pred['image'], pred['flow'], image, flow))
                loss_ftn.weight = tmp_weight
        idx = int(item['data_source_idx'].mode().values.item())
        data_source = data_sources[idx]
        losses = {f'{k}/{data_source}': mean(v) for k, v in losses.items()}
        losses['loss'] = sum(losses.values())
        losses[f'loss/{data_source}'] = losses['loss']

        return losses

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.valid_only:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                return {'val_' + k: v for k, v in val_log.items()}
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence)
            loss = losses['loss']
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for k, v in losses.items():
                self.train_metrics.update(k, v.item())

            if batch_idx % self.log_step == 0:
                msg = 'Train Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx < self.num_previews and (epoch - 1) % self.save_period == 0:
                with torch.no_grad():
                    self.preview(sequence, epoch, tag_prefix=f'train_{batch_idx}')

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        print("validation")
        if self.do_validation and (epoch - 1) % self.save_period == 0:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.true_once = False
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        i = 0
        for batch_idx, sequence in enumerate(self.valid_data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence, all_losses=True)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            for k, v in losses.items():
                self.valid_metrics.update(k, v.item())

            if batch_idx % self.val_log_step == 0:
                msg = 'Valid Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.valid_data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx in self.val_preview_indices and (epoch - 1) % self.save_period == 0:
                self.preview(sequence, epoch, tag_prefix=f'val_{i}')
                i += 1

        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)

    def preview(self, sequence, epoch, tag_prefix=''):
        """
        Plot visualisation to tensorboard.
        Plots input, output, groundtruth histograms and movies
        """
        print(f'Making preview {tag_prefix}')
        positive_event_previews, negative_event_previews, positive_voxels, negative_voxels, \
        pred_flows, pred_intensities, pred_aolps, pred_dolps, \
        flows, intensities, aolps, dolps = [], [], [], [], [], [], [], [], [], [], [], []
        # self.model.reset_states()
        # self.model.reset_states_90()
        # self.model.reset_states_45()
        # self.model.reset_states_135()
        # self.model.reset_states_0()
        # self.model.reset_states_s0()
        # self.model.reset_states_s1()
        # self.model.reset_states_s2()
        self.model.reset_states_i()
        self.model.reset_states_a()
        self.model.reset_states_d()
        # self.model.firenet_i90.reset_states()
        # self.model.firenet_i45.reset_states()
        # self.model.firenet_i135.reset_states()
        # self.model.firenet_i0.reset_states()
        # self.model.reset_states_i_shared()
        # self.model.reset_states_i2()
        # self.model.reset_states_i4()
        # self.model.reset_states_i8()
        # self.model.reset_states_a2()
        # self.model.reset_states_a4()
        # self.model.reset_states_a8()
        # self.model.reset_states_d2()
        # self.model.reset_states_d4()
        # self.model.reset_states_d8()
        for i, item in enumerate(sequence):
            item = {k: v[0:1, ...] for k, v in item.items()}  # set batch size to 1
            events, intensity, aolp, dolp, flow = self.to_device(item)
            # pred = self.model(events)

            # i90 = frame[:, :, 0::2, 0::2]
            # i45 = frame[:, :, 0::2, 1::2]
            # i135 = frame[:, :, 1::2, 0::2]
            # i0 = frame[:, :, 1::2, 1::2]
            #
            # intensity = i90
            # aolp = i45
            # dolp = i135

            pred = self.model(events)

            positive_event_previews.append(torch.sum(events[:, 0:events.shape[1] // 2, :, :], dim=1, keepdim=True))
            negative_event_previews.append(torch.sum(events[:, events.shape[1] // 2:-1, :, :], dim=1, keepdim=True))
            positive_voxels.append(events[:, 0:events.shape[1] // 2, :, :])
            negative_voxels.append(events[:, events.shape[1] // 2:-1, :, :])

            pred_flows.append(pred.get('flow', 0 * flow))
            pred_intensities.append(pred['i'])
            pred_aolps.append(pred['a'])
            pred_dolps.append(pred['d'])

            flows.append(flow)
            intensities.append(intensity)
            aolps.append(aolp)
            dolps.append(dolp)

        tc_loss_ftn = self.get_loss_ftn('temporal_consistency_loss')
        if self.true_once and tc_loss_ftn is not None:
            for i, intensity in enumerate(intensities):
                output = tc_loss_ftn(i, intensity, pred_intensities[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_intensity/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break
            for i, aolp in enumerate(aolps):
                output = tc_loss_ftn(i, aolp, pred_aolps[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_aolp/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break
            for i, dolp in enumerate(dolps):
                output = tc_loss_ftn(i, dolp, pred_dolps[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_dolp/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break

        vw_loss_ftn = self.get_loss_ftn('voxel_warp_flow_loss')
        if self.true_once and vw_loss_ftn is not None:
            for i, intensity in enumerate(intensities):
                output = vw_loss_ftn(positive_voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_positive_voxels/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break
            for i, intensity in enumerate(intensities):
                output = vw_loss_ftn(negative_voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_negative_voxels/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break

        # histogram
        self.writer.add_histogram(f'{tag_prefix}_flow/groundtruth', torch.stack(flows))
        self.writer.add_histogram(f'{tag_prefix}_intensity/groundtruth', torch.stack(intensities))
        self.writer.add_histogram(f'{tag_prefix}_aolp/groundtruth', torch.stack(aolps))
        self.writer.add_histogram(f'{tag_prefix}_dolp/groundtruth', torch.stack(dolps))

        positive_non_zero_voxel = torch.stack([s['events'][:, 0:s['events'].shape[1] // 2, :, :] for s in sequence])
        positive_non_zero_voxel = positive_non_zero_voxel[positive_non_zero_voxel != 0]
        if torch.numel(positive_non_zero_voxel) == 0:
            positive_non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_input/positive', positive_non_zero_voxel)

        negative_non_zero_voxel = torch.stack([s['events'][:, s['events'].shape[1] // 2:-1, :, :] for s in sequence])
        negative_non_zero_voxel = negative_non_zero_voxel[negative_non_zero_voxel != 0]
        if torch.numel(negative_non_zero_voxel) == 0:
            negative_non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_input/negative', negative_non_zero_voxel)
        self.writer.add_histogram(f'{tag_prefix}_flow/prediction', torch.stack(pred_flows))
        self.writer.add_histogram(f'{tag_prefix}_intensity/prediction', torch.stack(pred_intensities))
        self.writer.add_histogram(f'{tag_prefix}_aolp/prediction', torch.stack(pred_aolps))
        self.writer.add_histogram(f'{tag_prefix}_dolp/prediction', torch.stack(pred_dolps))

        video_tensor = make_flow_movie_p(positive_event_previews, negative_event_previews,
                                         pred_intensities, intensities, pred_aolps, aolps, pred_dolps, dolps,
                                         pred_flows, flows)
        self.writer.writer.add_video(f'{tag_prefix}', video_tensor, global_step=epoch, fps=20)

    def get_loss_ftn(self, loss_name):
        for loss_ftn in self.loss_ftns:
            if loss_ftn.__class__.__name__ == loss_name:
                return loss_ftn
        return None


class Trainer_RP(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(len(data_loader) // 100, 1)
        self.val_log_step = max(len(valid_data_loader) // 100, 1)

        mt_keys = ['loss']
        for data_source in data_sources:
            mt_keys.append(f'loss/{data_source}')
            for l in self.loss_ftns:
                mt_keys.append(f'{l.__class__.__name__}/{data_source}')
                for p in ['i90', 'i45', 'i135', 'i0', 'i', 'a', 'd']:
                    mt_keys.append(f'{p}/{l.__class__.__name__}/{data_source}')
        self.train_metrics = MetricTracker(*mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(*mt_keys, writer=self.writer)

        self.num_previews = config['trainer']['num_previews']
        self.val_num_previews = config['trainer'].get('val_num_previews', self.num_previews)
        self.val_preview_indices = select_evenly_spaced_elements(self.val_num_previews, len(self.valid_data_loader))
        self.valid_only = config['trainer'].get('valid_only', False)
        self.true_once = True  # True at init, turns False at end of _train_epoch

    def to_device(self, item):
        events = item['events'].float().to(self.device)
        raw = item['raw'].float().to(self.device)
        intensity = item['intensity'].float().to(self.device)
        aolp = item['aolp'].float().to(self.device)
        dolp = item['dolp'].float().to(self.device)
        flow = None if item['flow'] is None else item['flow'].float().to(self.device)

        return events, raw, intensity, aolp, dolp, flow

    def forward_sequence(self, sequence, all_losses=False):
        losses = collections.defaultdict(list)
        # self.model.reset_states()
        # self.model.reset_states_i90()
        # self.model.reset_states_i45()
        # self.model.reset_states_i135()
        # self.model.reset_states_i0()
        # self.model.reset_states_s0()
        # self.model.reset_states_s1()
        # self.model.reset_states_s2()
        self.model.reset_states_i()
        self.model.reset_states_a()
        self.model.reset_states_d()
        self.model.reset_states_i_shared()
        for i, item in enumerate(sequence):
            # get raw and polarization from dataloader directly
            events, raw, intensity, aolp, dolp, flow = self.to_device(item)
            i90 = raw[:, :, 0::2, 0::2]
            i45 = raw[:, :, 0::2, 1::2]
            i135 = raw[:, :, 1::2, 0::2]
            i0 = raw[:, :, 1::2, 1::2]

            pred = self.model(events)

            for loss_ftn in self.loss_ftns:
                loss_name = loss_ftn.__class__.__name__
                tmp_weight = loss_ftn.weight
                if all_losses:
                    loss_ftn.weight = 1.0
                if loss_name == 'perceptual_loss':
                    losses[f'i90/{loss_name}'].append(loss_ftn(pred['i90'], i90, normalize=True))
                    losses[f'i45/{loss_name}'].append(loss_ftn(pred['i45'], i45, normalize=True))
                    losses[f'i135/{loss_name}'].append(loss_ftn(pred['i135'], i135, normalize=True))
                    losses[f'i0/{loss_name}'].append(loss_ftn(pred['i0'], i0, normalize=True))
                    losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity, normalize=True))
                    losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp, normalize=True))
                    losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp, normalize=True))
                if loss_name == 'mse_loss':
                    losses[f'i90/{loss_name}'].append(loss_ftn(pred['i90'], i90))
                    losses[f'i45/{loss_name}'].append(loss_ftn(pred['i45'], i45))
                    losses[f'i135/{loss_name}'].append(loss_ftn(pred['i135'], i135))
                    losses[f'i0/{loss_name}'].append(loss_ftn(pred['i0'], i0))
                    losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                    losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                    losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                if loss_name == 'abs_sin_loss':
                    losses[f'i90/{loss_name}'].append(loss_ftn(pred['i90'], i90))
                    losses[f'i45/{loss_name}'].append(loss_ftn(pred['i45'], i45))
                    losses[f'i135/{loss_name}'].append(loss_ftn(pred['i135'], i135))
                    losses[f'i0/{loss_name}'].append(loss_ftn(pred['i0'], i0))
                    losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                    losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                    losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                if loss_name == 'ssim_loss':
                    losses[f'i90/{loss_name}'].append(loss_ftn(pred['i90'], i90))
                    losses[f'i45/{loss_name}'].append(loss_ftn(pred['i45'], i45))
                    losses[f'i135/{loss_name}'].append(loss_ftn(pred['i135'], i135))
                    losses[f'i0/{loss_name}'].append(loss_ftn(pred['i0'], i0))
                    losses[f'i/{loss_name}'].append(loss_ftn(pred['i'], intensity))
                    losses[f'a/{loss_name}'].append(loss_ftn(pred['a'], aolp))
                    losses[f'd/{loss_name}'].append(loss_ftn(pred['d'], dolp))
                loss_ftn.weight = tmp_weight
        idx = int(item['data_source_idx'].mode().values.item())
        data_source = data_sources[idx]
        losses = {f'{k}/{data_source}': mean(v) for k, v in losses.items()}
        losses['loss'] = sum(losses.values())
        losses[f'loss/{data_source}'] = losses['loss']

        return losses

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.valid_only:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                return {'val_' + k: v for k, v in val_log.items()}
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence)
            loss = losses['loss']
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for k, v in losses.items():
                self.train_metrics.update(k, v.item())

            if batch_idx % self.log_step == 0:
                msg = 'Train Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx < self.num_previews and (epoch - 1) % self.save_period == 0:
                with torch.no_grad():
                    self.preview(sequence, epoch, tag_prefix=f'train_{batch_idx}')

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        print("validation")
        if self.do_validation and epoch % 10 == 0:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.true_once = False
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        i = 0
        for batch_idx, sequence in enumerate(self.valid_data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence, all_losses=True)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            for k, v in losses.items():
                self.valid_metrics.update(k, v.item())

            if batch_idx % self.val_log_step == 0:
                msg = 'Valid Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.valid_data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx in self.val_preview_indices and (epoch - 1) % self.save_period == 0:
                self.preview(sequence, epoch, tag_prefix=f'val_{i}')
                i += 1

        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)

    def preview(self, sequence, epoch, tag_prefix=''):
        """
        Plot visualisation to tensorboard.
        Plots input, output, groundtruth histograms and movies
        """
        print(f'Making preview {tag_prefix}')
        positive_event_previews, negative_event_previews, positive_voxels, negative_voxels, \
        pred_flows, pred_intensities, pred_aolps, pred_dolps, \
        flows, intensities, aolps, dolps = [], [], [], [], [], [], [], [], [], [], [], []
        # self.model.reset_states()
        # self.model.reset_states_i90()
        # self.model.reset_states_i45()
        # self.model.reset_states_i135()
        # self.model.reset_states_i0()
        # self.model.reset_states_s0()
        # self.model.reset_states_s1()
        # self.model.reset_states_s2()
        self.model.reset_states_i()
        self.model.reset_states_a()
        self.model.reset_states_d()
        self.model.reset_states_i_shared()
        # self.model.reset_states_i2()
        # self.model.reset_states_i4()
        # self.model.reset_states_i8()
        # self.model.reset_states_a2()
        # self.model.reset_states_a4()
        # self.model.reset_states_a8()
        # self.model.reset_states_d2()
        # self.model.reset_states_d4()
        # self.model.reset_states_d8()
        for i, item in enumerate(sequence):
            item = {k: v[0:1, ...] for k, v in item.items()}  # set batch size to 1
            events, raw, intensity, aolp, dolp, flow = self.to_device(item)
            # pred = self.model(events)

            # i90 = frame[:, :, 0::2, 0::2]
            # i45 = frame[:, :, 0::2, 1::2]
            # i135 = frame[:, :, 1::2, 0::2]
            # i0 = frame[:, :, 1::2, 1::2]
            #
            # intensity = i90
            # aolp = i45
            # dolp = i135

            pred = self.model(events)

            positive_event_previews.append(torch.sum(events[:, 0:events.shape[1] // 2, :, :], dim=1, keepdim=True))
            negative_event_previews.append(torch.sum(events[:, events.shape[1] // 2:-1, :, :], dim=1, keepdim=True))
            positive_voxels.append(events[:, 0:events.shape[1] // 2, :, :])
            negative_voxels.append(events[:, events.shape[1] // 2:-1, :, :])

            pred_flows.append(pred.get('flow', 0 * flow))
            pred_intensities.append(pred['i'])
            pred_aolps.append(pred['a'])
            pred_dolps.append(pred['d'])

            flows.append(flow)
            intensities.append(intensity)
            aolps.append(aolp)
            dolps.append(dolp)

        tc_loss_ftn = self.get_loss_ftn('temporal_consistency_loss')
        if self.true_once and tc_loss_ftn is not None:
            for i, intensity in enumerate(intensities):
                output = tc_loss_ftn(i, intensity, pred_intensities[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_intensity/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break
            for i, aolp in enumerate(aolps):
                output = tc_loss_ftn(i, aolp, pred_aolps[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_aolp/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break
            for i, dolp in enumerate(dolps):
                output = tc_loss_ftn(i, dolp, pred_dolps[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_dolp/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=2)
                    break

        vw_loss_ftn = self.get_loss_ftn('voxel_warp_flow_loss')
        if self.true_once and vw_loss_ftn is not None:
            for i, intensity in enumerate(intensities):
                output = vw_loss_ftn(positive_voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_positive_voxels/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break
            for i, intensity in enumerate(intensities):
                output = vw_loss_ftn(negative_voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_negative_voxels/tc_{tag_prefix}',
                                                 video_tensor, global_step=epoch, fps=1)
                    break

        # histogram
        self.writer.add_histogram(f'{tag_prefix}_flow/groundtruth', torch.stack(flows))
        self.writer.add_histogram(f'{tag_prefix}_intensity/groundtruth', torch.stack(intensities))
        self.writer.add_histogram(f'{tag_prefix}_aolp/groundtruth', torch.stack(aolps))
        self.writer.add_histogram(f'{tag_prefix}_dolp/groundtruth', torch.stack(dolps))

        positive_non_zero_voxel = torch.stack([s['events'][:, 0:s['events'].shape[1] // 2, :, :] for s in sequence])
        positive_non_zero_voxel = positive_non_zero_voxel[positive_non_zero_voxel != 0]
        if torch.numel(positive_non_zero_voxel) == 0:
            positive_non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_input/positive', positive_non_zero_voxel)

        negative_non_zero_voxel = torch.stack([s['events'][:, s['events'].shape[1] // 2:-1, :, :] for s in sequence])
        negative_non_zero_voxel = negative_non_zero_voxel[negative_non_zero_voxel != 0]
        if torch.numel(negative_non_zero_voxel) == 0:
            negative_non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_input/negative', negative_non_zero_voxel)

        self.writer.add_histogram(f'{tag_prefix}_flow/prediction', torch.stack(pred_flows))
        self.writer.add_histogram(f'{tag_prefix}_intensity/prediction', torch.stack(pred_intensities))
        self.writer.add_histogram(f'{tag_prefix}_aolp/prediction', torch.stack(pred_aolps))
        self.writer.add_histogram(f'{tag_prefix}_dolp/prediction', torch.stack(pred_dolps))

        video_tensor = make_flow_movie_p(positive_event_previews, negative_event_previews,
                                         pred_intensities, intensities, pred_aolps, aolps, pred_dolps, dolps,
                                         pred_flows, flows)
        self.writer.writer.add_video(f'{tag_prefix}', video_tensor, global_step=epoch, fps=20)

    def get_loss_ftn(self, loss_name):
        for loss_ftn in self.loss_ftns:
            if loss_ftn.__class__.__name__ == loss_name:
                return loss_ftn
        return None
