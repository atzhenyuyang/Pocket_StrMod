import os
import time
from torch.utils import tensorboard
import torch
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda import amp



def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def timewait(time_gap):
    d = time_gap//(24*3600)
    d_h = time_gap%(24*3600)
    h = d_h//3600
    h_m = d_h%3600
    m = h_m//60
    s = h_m%60
    if d > 0:
        out = '{}d {}h {}m {}s'.format(int(d),int(h),int(m),round(s,2))
    elif h > 0:
        out = '{}h {}m {}s'.format(int(h),int(m),round(s,2))
    elif m > 0:
        out = '{}m {}s'.format(int(m),round(s,2))
    else:
        out = '{}s'.format(round(s,2))
    return out

def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

class Experiment(object):
    def __init__(self, model, train_set, optimizer, scheduler=None, device='cuda',
                 valid_set=None, clip_grad=True, max_norm=5, norm_type=2, grad_accu_step=8,
                 pos_noise_std=0.1, data_parallel=False, use_amp=False):
        if data_parallel:
            self.model = pyg.nn.DataParallel(model)
        else:
            self.model = model#.to(device)
        if 'config' in model.__dict__:
            self.model_config = model.config
        else:
            self.model_config = None
        self.train_set = train_set
        self.valid_set = valid_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_train_data = len(train_set)
        self.data_parallel = data_parallel
        self.grad_accu_step = grad_accu_step
        if valid_set:
            self.num_valid_data = len(valid_set)
        else:
            self.num_valid_data = None
        self.clip_grad = clip_grad
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.with_tb = False
        self.device = device
        self.pos_noise_std = pos_noise_std
        self.use_amp = use_amp
        if self.use_amp:
            self.grad_scaler = amp.GradScaler()

    @staticmethod
    def get_log(out_dict, key_word, it, time_gap=None):
        log = []
        for key, value in out_dict.items():
            log.append(' {}:{:.5f} |'.format(key, value))
        log.insert(0, '[{} {}]'.format(key_word, it))
        if time_gap:
            log.append(' Time: {}'.format(time_gap))
        return ''.join(log)

    @staticmethod
    def write_summary(out_dict, writer, key_word, num_iter, scheduler=None, optimizer=None):
        for key, value in out_dict.items():
            writer.add_scalar('{}/{}'.format(key_word, key), value, num_iter)
        if scheduler is not None:
            writer.add_scalar('{}/lr'.format(key_word), optimizer.param_groups[0]['lr'], num_iter)
        writer.flush()

    @staticmethod
    def get_num_iter(num_data, batch_size):
        if num_data % batch_size == 0:
            n_iter = int(num_data/batch_size)
        else:
            n_iter = int(num_data/batch_size) + 1
        return n_iter
    
    @property
    def parameter_number(self):
        return get_parameter_number(self.model)
    
    def _train_step(self, batch, it=0, print_log=False):
        self.model.train()
        cpx_noise = torch.randn_like(batch.cpx_pos) * self.pos_noise_std
        batch.cpx_pos = batch.cpx_pos + cpx_noise
        out_dict = self.model.get_loss(batch)
        out_dict = {k:v/self.grad_accu_step for k,v in out_dict.items()}
        out_dict['loss'].backward()
        for k,v in out_dict.items():
            if k in self.AccuOutDict:
                self.AccuOutDict[k] += v.item()
            else:
                self.AccuOutDict[k] = v.item()
        
        if (it % self.grad_accu_step) == 0:
            orig_grad_norm = clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.max_norm, 
                    norm_type=self.norm_type,
                    error_if_nonfinite=True
                    )   
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer.add_scalar('train/step/grad', orig_grad_norm, it//self.grad_accu_step)
            self.write_summary(self.AccuOutDict, self.writer, 'train/step', it//self.grad_accu_step)
            end = time.time()
            time_gap = end - self.start_time
            log = self.get_log(self.AccuOutDict, 'Step', it//self.grad_accu_step, time_gap='{:.3f}'.format(time_gap))
            with open(self.logdir+'training.log', 'a') as log_writer:
                log_writer.write(log + '\n')
            if print_log:
                print(log)
            self.start_time = time.time()
            self.AccuOutDict = {}
        return self.AccuOutDict
    
    def train_epoch(self, n_iter, n_epoch, print_log=False):
        start = time.time()
        log_dict = {}
        for i in range(n_iter):
            if self.data_parallel:
                batch = next(self.train_loader).cuda()
            else:
                batch = next(self.train_loader).to(self.device)
            compose_noise = torch.randn_like(batch.compose_pos) * self.pos_noise_std
            batch.compose_pos = batch.compose_pos + compose_noise
            out_dict = self._train_step(batch, it=i+n_epoch*n_iter, print_log=print_log)
            for key, value in out_dict.items():
                if key not in log_dict:
                    log_dict[key] = value if 'acc' in key else value * batch.num_graphs
                else:
                    log_dict[key] += value if 'acc' in key else value * batch.num_graphs
        for key, value in log_dict.items():
            if 'acc' in key:
                log_dict[key] = value / n_iter
            else:
                log_dict[key] = value / self.num_train_data

        if self.with_tb:
            self.write_summary(log_dict, self.writer, 'train/epoch', n_epoch, 
                               scheduler=self.scheduler, optimizer=self.optimizer)
        end = time.time()
        time_gap = timewait(end - start)
        log = self.get_log(log_dict, 'Epoch', n_epoch, time_gap=time_gap)
        with open(self.logdir+'training.log', 'a') as log_writer:
            log_writer.write(log + '\n')
        if print_log:
            print(log)
        

    def validate(self, n_iter, n_epoch, print_log=False, schedule_key='loss'):
        start = time.time()
        log_dict = {}
        with torch.no_grad():
            self.model.eval()
            for _ in range(n_iter):
                batch = next(self.valid_loader).to(self.device)
                out_dict = self.model.get_loss(batch)
                for key, value in out_dict.items():
                    if key not in log_dict:
                        log_dict[key] = value if 'acc' in key else value * batch.num_graphs
                    else:
                        log_dict[key] += value if 'acc' in key else value * batch.num_graphs

        for key, value in log_dict.items():
            if 'acc' in key:
                log_dict[key] = value / n_iter
            else:
                log_dict[key] = value / self.num_valid_data
        if self.scheduler:
            self.scheduler.step(log_dict[schedule_key])
        if self.with_tb:
            self.write_summary(log_dict, self.writer, 'val/epoch', n_epoch, 
                               scheduler=self.scheduler, optimizer=self.optimizer)
        end = time.time()
        time_gap = timewait(end - start)
        log = self.get_log(log_dict, 'Validate', n_epoch, time_gap=time_gap)
        with open(self.logdir+'training.log', 'a') as log_writer:
            log_writer.write(log + '\n')
        if print_log:
            print(log)
        return log_dict['loss']
    
    
    def fit_step(self, num_step, valid_per_step=5000, train_batch_size=1, valid_batch_size=16, print_log=True,
                 with_tb=True, logdir='./training_log', schedule_key='loss', num_workers=0, pin_memory=False, follow_batch=[], exclude_keys=[], max_edge_num_in_batch=900000):
        
        self.train_loader = inf_iterator(DataLoader(
            self.train_set, 
            batch_size = train_batch_size, 
            shuffle = True,
            num_workers = num_workers,
            pin_memory = pin_memory,
        ))
        self.train_batch_size = train_batch_size

        if self.valid_set:
            self.valid_loader = inf_iterator(DataLoader(
                self.valid_set, 
                batch_size = valid_batch_size, 
                shuffle = False,
                num_workers = num_workers,
                pin_memory = pin_memory,
            ))
            self.valid_batch_size = valid_batch_size
            
        self.n_iter_train = self.get_num_iter(self.num_train_data, self.train_batch_size)
        if self.num_valid_data:
            self.n_iter_valid = self.get_num_iter(self.num_valid_data, self.valid_batch_size)
        date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.logdir = logdir+'/'+date+'/'
        verify_dir_exists(self.logdir)
        open(self.logdir+'model_config.dir','w').write(str(self.model_config))
        self.with_tb = with_tb
        if self.with_tb:
            self.writer = tensorboard.SummaryWriter(self.logdir)
        log_writer = open(self.logdir+'training.log', 'w')
        log_writer.write('\n######## {}; batch_size {} ########\n'.format(self.parameter_number, train_batch_size))
        log_writer.close()

        if print_log:
            print('\n######## {} ########\n'.format(self.parameter_number))

        self.AccuOutDict = {}
        self.start_time = time.time()
        step = 1
        step_valid = 0
        current_step_valid = 1
        while step < num_step+1:
            batch = next(self.train_loader).to(self.device)
            if batch.cpx_edge_index.size(1) > max_edge_num_in_batch:
                print(batch.cpx_edge_index.size(1),'>',max_edge_num_in_batch, ' skip')
                continue
            cpx_noise = torch.randn_like(batch.cpx_pos) * self.pos_noise_std
            batch.cpx_pos = batch.cpx_pos + cpx_noise
            self._train_step(batch, it=step, print_log=print_log)
            if (step_valid%valid_per_step==0 or step_valid==num_step) and step_valid!=0:
                if current_step_valid != step_valid:
                    if self.num_valid_data:
                        val_loss = self.validate(self.n_iter_valid, step_valid, schedule_key=schedule_key, print_log=print_log)
                    ckpt_path = self.logdir + '/ckpt/'
                    verify_dir_exists(ckpt_path)
                    torch.save({
                        'config': self.model_config,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'iteration': step_valid, 
                    }, ckpt_path+'/{}.pt'.format(step_valid))
                    current_step_valid = step_valid
            if step % self.grad_accu_step == 0:
                step_valid += 1
            step += 1
            
           
