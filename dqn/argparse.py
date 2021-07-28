#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:12:04 2021

@author: monikkabusto
"""

import argparse

parser = argparse.ArgumentParser(description='')

#optional
parser.add_argument('-argument', type=dtype, default=x, help = 'helpstring')

#required
parser.add_argument('argument', type=dtype, default=x, help = 'helpstring')

args = parser.parse_args()

variable = args.argument

