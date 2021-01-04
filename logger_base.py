#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   logger_base.py
@Time    :   2020/12/12 22:30:21
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib
import functools
import logging
 
def create_logger():
  logger = logging.getLogger("test_log")
  logger.setLevel(logging.INFO)
  fh = logging.FileHandler("test.log")
  fmt = "\n[%(asctime)s-%(name)s-%(levelname)s]: %(message)s"
  formatter = logging.Formatter(fmt)
  fh.setFormatter(formatter)
  logger.addHandler(fh) 
  return logger
 
def log_exception(fn):
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    print('arg',*args, "kwarg",**kwargs)
 
    logger = create_logger()
    try:
        if kwargs:
            fn(*args, **kwargs)
        else:
            fn(*args)
    except Exception as e:
        logger.exception("[Error in {}] msg: {}".format(__name__, str(e)))
        raise
  print('fn',fn)
  return wrapper