"""
Main CLI entry point for the Flamino CLI 
"""


import click

from flamino.cli.data import data
from flamino.cli.train import train


@click.group()
def flamino():
    pass


flamino.add_command(data)
flamino.add_command(train)

if __name__ == '__main__':
    flamino()
