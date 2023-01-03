import logging
import torch
import torch.nn as nn
import pytest
from torchsummary import summary
import sys

from models.transformer import TransformerModel, BaseEncoder
from utils import parse_cmd_args

# Basic tests for the argparser


class TestParseCmdArgs:

    def setup_method(self):
        # This replaces the pytest command line args so we can test properly
        sys.argv[1:] = []
        self.opts = parse_cmd_args()

    def teardown_method(self):
        pass

    def test_default_values(self):
        assert self.opts.model.name == "Transformer"
        assert self.opts.model.n_hid == 32
        assert self.opts.model.emb_dim == 32
        assert self.opts.model.n_token == 256
        assert self.opts.model.n_head == 4
        assert self.opts.model.dropout == 0.1
        assert self.opts.model.learning_rate == 0.001
        assert self.opts.model.epochs == 10

        assert type(self.opts.model.name) == str
        assert type(self.opts.model.n_hid) == int
        assert type(self.opts.model.emb_dim) == int
        assert type(self.opts.model.n_token) == int
        assert type(self.opts.model.n_head) == int
        assert type(self.opts.model.dropout) == float
        assert type(self.opts.model.learning_rate) == float
        assert type(self.opts.model.epochs) == int

    def update_args(self, arg_list):
        sys.argv[1:] = arg_list
        self.opts = parse_cmd_args()

    def test_updated_args(self):
        arg_list = ["--model.epochs=10", "--model.n_layers=12"]
        self.update_args(arg_list)
        assert self.opts.model.epochs == 10
        assert self.opts.model.n_layers == 12

    def test_bad_args(self):
        arg_list = ["--Model.epochs=10", "--model.n_layers=12"]
        try:
            self.update_args(arg_list)
        except SystemExit as error:
            # TODO: Check this is actually being run
            assert type(error) == SystemExit
        
        

    # TODO: Add a test for args and wandb as well - may be a seperate test location
