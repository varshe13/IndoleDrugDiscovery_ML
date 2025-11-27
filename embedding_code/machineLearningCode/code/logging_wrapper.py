"""
Description: Logging wrapper to create a logger printing to screen  and saving to a file at the same time

Default settings:   console logging level = DEBUG
                    file logging level = INFO
                    logging format is time - name - level - message
    
11/21/2023 11:08PM, BC

11/23/2023 11AM, BC
    change logging format. Remove time from the formatter.

12/10/2023 10PM, BC
    Added logging for each GPU.
    Only print GPU0 info.
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import time
import os

# 6 default levels (0,10,20,30,40,50): NOTSET DEBUG, INFO, WARNING, ERROR, CRITICAL



def now():
    return time.strftime("%Y%m%dT%H%M%S%z")

class Logging_Wrapper:
    # addLoggingLevel('TRACE', logging.INFO + 5)
    # logging.getLogger(__name__).setLevel("TRACE")

   
    def __init__(self, logfilename):    
        # self.formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
        self.formatter = logging.Formatter("%(name)s — %(levelname)s \n%(message)s")
        self.datefmt = "%Y%m%dT%H%M%S%z"
        self.logfilename = logfilename
        logging.basicConfig(level=logging.INFO)

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(self.formatter)
        console_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        return console_handler
    
    def get_file_handler(self):
        file_handler = logging.FileHandler(self.logfilename)
        file_handler.setLevel(logging.INFO)   #set level
        file_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        return file_handler
    

    def get_file_console_logger(self, name=__name__):
        """ 
        arg:  name -- use the current running filename as logger name
                      default is __name__, the module name
        """
        logger = logging.getLogger(name)
        # if int(os.environ["RANK"]) == 0:
        #     logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler())
        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False
        return logger
    
# out of the class    
def logger_console_file(logpath, filename):
    """ create a logger using the info in opt
        logger's name is opt.name_timestamp.log
        Default: DEBUG for console. INFO for file
        11/22/2023 4pm, BC
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = os.path.join(logpath, filename+'_'+ timestamp+'.log')
    L =  Logging_Wrapper(log_file_name)
    logger = L.get_file_console_logger()
    return logger


class Logging_GPUs:

    def __init__(self, path, world_size:int):
        """ Create loggers for each integer in ranks.
            Args:
                path = the filename stem. str
                world_size = an int of the total processes. 
            Output:
                create loggers for each rank in DDP
            Example:
                path = "~/Log/sorting_obj_1210"
                world_size = 3
                Four loggers (len(ranks)) will be created.
                Each logger saves logging output to a file. 
                File names = "sorting_obj_1210_GPU0.log"
                            "sorting_obj_1210_GPU1.log"
                            "sorting_obj_1210_GPU2.log"
            Important: Logger should have its own name. Same logger with different file handles will log to all files.
        """
        self.loggers = []
        for i in range(world_size):
            fname = path + '_GPU' + str(i) + ".log"
            L=Logging_Wrapper(fname)
            # important: give logger a differnt name to log to a different file
            self.loggers.append(L.get_file_console_logger(name= f"GPU{i}"))
            

    def info(self, msg):
        rank = int(os.environ["RANK"])
        self.loggers[rank].info(msg)






if __name__ == "__main__":


    ## test Logging_GPUs
    path = "./testgpu"
    worldsize = 3
    loggers = Logging_GPUs(path=path, world_size=worldsize)


    os.environ["RANK"] = "2"
    loggers.info("write to rank 2")

    os.environ["RANK"] = "1"
    loggers.info("write to rank 111.")

    os.environ["RANK"] ="0"
    loggers.info("write to rank 000.")

  