"""
Train and Test four classes 0, 0.5, 1.0 1.5 --> 0,1,2,3 as non, low, med and high

12/07/2023, 9PM, BC
   Finding: "low" activity category has 0 predictions.  
     No similar behaviour in 3 classes (0,0.5) as low, l as med and 1.5 as high

12/08/2023, 3PM, BC
    Add normailze and change to 3 classes
    Add scheduler

12./08/2023, 4PM, BC
    same as train3.py
    Add ddp

3/11/2024, 1pm, BC
   change dataset to 122623_all_GP_molecules_4D.pickle

"""
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn as nn
# DDP
from torch.nn.parallel import DistributedDataParallel  as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

# import torchvision.transforms as transforms
import time
import os
import sys


from models.cnn3d_indole import cnn3d, cnn3d_2
from logging_wrapper import logger_console_file, now, Logging_GPUs
from utils import  get_obj_attributes
from utils_torch import  find_free_port
# from utils_torch import ddp_setup

from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# DDP 
def ddp_setup(backend="nccl"):
    init_process_group(backend=backend)
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    
class Options():
    name="Indole4D"
    suffix=""
    model_name = "cnn3d"
    model_description = "cnn3d with normalization on each channel"
    dataset_name = "122623_all_GP_molecules_4D"
    dataset_description = "2800 moledules"
    root_data_dir="../Data4D"
    fname_indole = "122623_all_GP_molecules_4D.pickle"
    fname_act = "122623_all_moleducles_GP_activities.pickle"
    result_dir="../Output/Results"
    checkpoint_dir="../Output/Checkpoints"
    logfile_dir = "../Output/Logs"
    random_seed = 42

    classes = ("Low", "Med", "High")

    epochs=300
    batch_size=256
    split_ratio = (0.80,0.20)
    workers = 8

    lr = 0.001
    max_lr = 0.001
    pin_memory=True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # DDP
    backend = "nccl"  
    MASTER_ADDR = "localhost"
    MASTER_PORT = find_free_port()  # find_free_port() returns a string
    WORLD_SIZE = "8"
    nnodes = 1 
    nproc_per_node = 8         
    # rdzv_id=100
    # rdzv_backend=c10d

    save_model=False

opt = Options()
# logger = logger_console_file(opt.logfile_dir, opt.name)
logpath = os.path.join(opt.logfile_dir, opt.name) + time.strftime("%Y%m%dT%H%M%S")
logger = Logging_GPUs(path=logpath, world_size=int(opt.WORLD_SIZE))
writer = SummaryWriter()


def read_pickle_file(fname):
    with open(fname, "rb") as f: # "rb" because we want to read in binary mode
        db = pickle.load(f)
    return db


class IndoleDataset(Dataset):
    """ Indole dataset
    """
    def __init__(self, indole_file, activity_file):
        """
        Args: indoles - 5D matrix in N, C, D1, D2, D3 =
             activities - 1D vector with N elements. The target.
        """
        self.indoles = self._get_indole5d(indole_file)
        self.activities = self._get_activities(activity_file)
        indole_mean, indole_std = self._normalize_indoles()
        # torch_print_all(self.activities)

        assert len(self.activities) == len(self.indoles), "indoles and activities should have the same number of elements."
        self.N = len(self.indoles)

    def __len__(self):
        return len(self.activities)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        indole = self.indoles[idx]
        activity = self.activities[idx]
        sample = {'indole': indole, 'activity': activity}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample
    
    def _get_indole5d(self, indole_file):
        indoles = read_pickle_file(indole_file)
        N = len(indoles)
        C,D1,D2,D3 = indoles[0].shape
        indoles5D = torch.zeros(N, C, D1, D2, D3)
        for i, vol in enumerate(indoles):
            indoles5D[i] =torch.from_numpy(vol)
        return indoles5D
    
    def _get_activities(self, activity_file):
        activities = read_pickle_file(activity_file)
        activities = torch.tensor(activities)
        activities[activities<0.1] = 0.5
        activities = (activities - 0.5) * 2 # convert 0.0, 0.5, 1, 1.5 to 0, 1,2
        return activities.type(torch.int64)
    
    def _normalize_indoles(self):
        """ for normalization"""
        mass = self.indoles[:,0]
        charge = self.indoles[:,1]
        m_mass = torch.mean(mass[mass>0])
        std_mass = torch.std(mass[mass>0])
        m_charge = torch.mean(charge[charge!=0])
        std_charge = torch.std(charge[charge!=0])

        self.indoles[:,0] = (self.indoles[:,0] - m_mass)/std_mass
        self.indoles[:,1] = (self.indoles[:,1] - m_charge)/std_charge

        return (m_mass, m_charge), (std_mass, std_charge)
        # print(f"  mean and std: {m_mass=}, {std_mass=}, {m_charge=}, {std_charge}")
        

#####################################
class TrainTest:

    def __init__(self, model, optimizer, criterion, opt):
        
        self.opt = opt

        # DDP
        # self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.gpu_id = int(os.environ["RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.optimizer = optimizer
        self.criterion = criterion.to(opt.device)
        # scheduler
        end_lr = opt.lr/5
        stages = 10
        # gamma = torch.pow(10, torch.log10(torch.tensor(end_lr/opt.lr))/stages)
        gamma = torch.pow(torch.tensor(end_lr/opt.lr), 1/stages)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.epochs//stages, gamma=gamma)

        f_indole = os.path.join(opt.root_data_dir, opt.fname_indole)
        f_act = os.path.join(opt.root_data_dir, opt.fname_act)
        Indole_dataset = IndoleDataset(f_indole, f_act)
        N = Indole_dataset.N    
        self.trainset, self.testset = torch.utils.data.random_split(Indole_dataset, [int(N*opt.split_ratio[0]), N - int(N*opt.split_ratio[0])],
                                                                    generator=torch.Generator().manual_seed(self.opt.random_seed))
        sampler=DistributedSampler(self.trainset, shuffle=True, seed=self.gpu_id)
        self.trainloader = DataLoader(self.trainset, batch_size=opt.batch_size, sampler=sampler, shuffle=False, num_workers=self.opt.workers)
        self.testloader = DataLoader(self.testset, batch_size=opt.batch_size, shuffle=False, num_workers=self.opt.workers)

        data_info = f"{now()}\n    --- Trainset size={len(self.trainset)}, Testset size={len(self.testset)}."
        logger.info(data_info)
    
    def train(self, epoch):
        start_time = time.time()
        batch_loss = 0

        # DDP
        # logger.info(f"      Train on [GPU{self.gpu_id}] Epoch={epoch} ")
        self.trainloader.sampler.set_epoch(epoch)
        for i, data in enumerate(self.trainloader, start=0):
            inputs = data['indole'].to(self.gpu_id)
            labels = data['activity'].to(self.gpu_id)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output, labels)
            # logger.info(f" In train: {output=}, {labels=}, {loss=}")
            # logger.info(f"     ==> lr = {optimizer.param_groups[0]['lr']}")
            loss.backward()
            self.optimizer.step()
            batch_loss +=loss.item()

        self.scheduler.step()
        batch_loss /=(i+1)
        logger.info(f"    [GPU{self.gpu_id}]: Train loss = {batch_loss:.6f}. time = {time.time() - start_time:.1f} seconds. lr={optimizer.param_groups[0]['lr']}")
        return batch_loss


    def test(self):
        start_time = time.time()
        batch_loss=0.0
        with torch.no_grad():
            for i, data in enumerate(self.testloader, start=0):
                inputs = data['indole'].to(self.gpu_id)
                labels = data['activity'].to(self.gpu_id)               
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                batch_loss +=loss.item()
                _, predicted = torch.max(output.data, 1)
            batch_loss /=(i+1)
            logger.info(f"    [GPU{self.gpu_id}]: Test loss = {batch_loss:.6f}. time = {time.time() - start_time:.1f} seconds.")
        return batch_loss, predicted


    def save_model(self, epoch, train_loss):
        
        tmp2_=f"{train_loss:.3E}"
        tmp2_=tmp2_.replace("E+", "E")
        fname = f"{self.opt.name}_epoch{epoch}_{time.strftime('%m%d%H%M')}_TL_{tmp2_}.pt"
        save_path = os.path.join(self.opt.checkpoint_dir, fname)

        if hasattr(self.model, 'module'):
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        logger.info(f"    [GPU{self.gpu_id}]:  Checkpoint: {save_path}")

        # barrier() # make sure 

        return save_path

    def run(self):

        best_train_loss = 1E20
        best_test_loss = 1E20

        for epoch in range(1, self.opt.epochs + 1):
            logger.info(f"{now()}:  Epoch {epoch}/{self.opt.epochs}:")
            train_loss = self.train(epoch)
            test_loss, _ = self.test()

            if train_loss < best_train_loss and test_loss<best_test_loss:
                best_train_loss = train_loss
                best_test_loss = test_loss
                pt_filename = self.save_model(epoch, train_loss)
                logger.info(f"   Checkpoint: {pt_filename}")
         
            writer.add_scalars('Loss', {'train_loss':train_loss, 'test_loss':test_loss}, epoch)

        # DDP. when training is done. out of for epoch loop
        destroy_process_group()


if __name__ =='__main__':

    #Removing logger info due to non-distributed launch
    # logger.info(f"   ==== {now()}.  {sys.argv}====\n")
    opt_dict, opt_str = get_obj_attributes(opt)
    # logger.info("   --- Configurations ---")
    # logger.info(opt_str)

    # model = get_model_resnet34(in_channels=2, n_classes=3)
    # model = get_model_effinetV4(in_channels=2, n_classes=3)
    model = cnn3d_2(in_channels=2, n_classes=len(opt.classes))
    # summary(model, input_size=(5, 2, 30,30,30))
    # print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    ddp_setup(backend="nccl")

    # for k,v in os.environ.items():
    #     print (f"{k}={v}")

    TVT = TrainTest(model=model, optimizer=optimizer, criterion=criterion, opt=opt )

    TVT.run()
    writer.close()

 