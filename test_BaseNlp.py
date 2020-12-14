#!/usr/bin/env python
# coding: utf-8
######################################################################
# test_Genom.py                                                       #
# Test the genom library                                             #
# License:  MIT 2019 Itai Danielli                                   #
######################################################################
import os
import sys
from unittest import TestCase


# Set project include path.
sys.path.append('./')
sys.path.append(os.path.join(os.getcwd(),'../'))
from PyBaseNlp import TextNltk





######################################################################
#                                                                    #
#                        Test the Genom Class                        #
#                                                                    #
######################################################################
class TestBaseNlpCls(TestCase):

    def test_NltkDistance(self):
        w1 = 'mapping'
        w2 = 'mappings'
        w3 = 'mappingx'

        print('Edit distance', TextNltk.EditDistance(w1, w2))
        print('Jacard distance', TextNltk.JacardDistance(w1,w2))

        test_prec, test_recl, F = TextNltk.PRF_Test(w1,w2)
        print("test_prec {}, test_recl {}, F {}".format(test_prec, test_recl, F))

        print(TextNltk.ConfMatrix(w2,w3))
        return



