#!usr/bin/python3
# -*- coding: utf-8 -*-

import os
import re
import pathlib
from subprocess import check_output as ck
import subprocess

from .src import Fore, Back, Style


class Execute:
    """Execute commands trough shell."""
    def __init__(self, cmd: str, path='/root', exe='/bin/bash', ack='') -> None:
        self.cmd = cmd
        self.path = pathlib.Path(path)
        self.exe = exe
        self.ack = ack

    def __call__(self) -> str:
        process = subprocess.Popen(
            self.cmd, cwd=self.path, shell=True, executable=self.exe, encoding='UTF-8',
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate(self.ack)
        return f"{stdout.strip()}\n{stderr.strip()}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cmd={self.cmd}, path={self.path}, exe={self.exe}, ack='{self.ack}')"

    def __str__(self) -> str:
        return f'{Fore.BLUE}┌──({Style.RESET_ALL}{Fore.RED}{ck("whoami", encoding="UTF-8").strip()}{Style.RESET_ALL}' \
               f'💀{Fore.RED}{ck("hostname", encoding="UTF-8").strip()}{Style.RESET_ALL}' \
               f'{Fore.BLUE})-[{Style.RESET_ALL}' \
               f'{Fore.WHITE}~{os.getcwd()}{Style.RESET_ALL}' \
               f'{Fore.BLUE}]\n└─{Style.RESET_ALL}{Fore.RED}#{Style.RESET_ALL} ' \
               f'{self.shell()}'

    def shell(self) -> str:
        text = self.cmd
        text = text.split()
        text = f'{Fore.LIGHTGREEN_EX}{text[0]}{Style.RESET_ALL} {Fore.LIGHTWHITE_EX}' \
               f'{" ".join(text[1:])}{Style.RESET_ALL}'
        return text


class Separator:
    def __init__(self, num: int, text: str) -> None:
        print(f'{Fore.BLACK}{Back.WHITE}{" " * num + f"{text}" + " " * num}{Style.RESET_ALL}')


class Print:
    """
    Print with colorama:
    colours = ['b', 'y', 'c', 'k', 'm' ,'w' ,'r' ,'g']
    usage: Print('b(Hello)')
    """
    def __init__(self, text: str) -> None:
        self.text = text
        self.__str__()

    def __str__(self) -> None:
        print(self.colour())

    def colour(self) -> str:
        pattern = r'\w+\((?<=\().+?(?=\))\)'
        for text in re.findall(pattern, self.text):
            pattern1 = r'([mkcwybrgml])(\()(.+)(\))'
            pattern2 = r'\1\2"\3", "{}"\4'.format(text)
            try:
                exec(f'self.{re.sub(pattern1, pattern2, text)}')
            except (SyntaxError, NameError, AttributeError, TypeError):
                pass
        return self.text

    def b(self, text: str, replace: str) -> None:
        self.text = self.text.replace(replace, f'{Fore.LIGHTBLUE_EX}{text}{Style.RESET_ALL}')

    def y(self, text: str, replace: str) -> None:
        self.text = self.text.replace(replace, f'{Fore.LIGHTYELLOW_EX}{text}{Style.RESET_ALL}')

    def c(self, text: str, replace: str) -> None:
        self.text = self.text.replace(replace, f'{Fore.LIGHTCYAN_EX}{text}{Style.RESET_ALL}')

    def k(self, text: str, replace: str) -> None:
        self.text = self.text.replace(replace, f'{Fore.LIGHTBLACK_EX}{text}{Style.RESET_ALL}')

    def m(self, text: str, replace: str) -> None:
        self.text = self.text.replace(replace, f'{Fore.LIGHTMAGENTA_EX}{text}{Style.RESET_ALL}')

    def w(self, text: str, replace: str) -> None:
        self.text = self.text.replace(replace, f'{Fore.LIGHTWHITE_EX}{text}{Style.RESET_ALL}')

    def r(self, text, replace):
        self.text = self.text.replace(replace, f'{Fore.LIGHTRED_EX}{text}{Style.RESET_ALL}')

    def g(self, text: str, replace: str) -> None:
        self.text = self.text.replace(replace, f'{Fore.LIGHTGREEN_EX}{text}{Style.RESET_ALL}')


def execute(cmd: str, run=True) -> None:
    cmd = Execute(cmd)
    print(cmd)
    if run:
        std = cmd()
        print(std)
