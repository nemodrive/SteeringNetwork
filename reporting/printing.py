from termcolor import colored as clr
import sys


class Printer(object):

    def __init__(self, name, color="blue", verbose=1):
        if hasattr(self, "name"):
            print("Already have something... " + self.name)
        self.name = name
        self.verbose = verbose
        self.color = color

    @staticmethod
    def __make_line(*args, **kwargs):
        args = [str(a) for a in args]
        kwargs = [f"{v['name']} = {clr(v['value'], *v.get('colors', ['red']))}"
                  for _, v in kwargs.items()]
        line = clr(" | ", "yellow").join(args + kwargs)
        return line

    def _print(self, *args, **kwargs):
        line = self.__make_line(*args, **kwargs)
        print(clr(f"[{self.name:s}] ", self.color) + line)

    def info(self, *args, **kwargs):
        if self.verbose > 0:
            self._print(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.verbose > 1:
            self._print(*args, **kwargs)

    # Writing to stdout (if verbose > 0)

    def rout(self, *args, **kwargs):
        if self.verbose > 0:
            line = self.__make_line(*args, **kwargs)
            sys.stdout.write("\r\033[K" + clr(f"[{self.name:s}] ", self.color) +
                             line)

    def nout(self, *args, **kwargs):
        if self.verbose > 0:
            line = self.__make_line(*args, **kwargs)
            sys.stdout.write("\n" + clr(f"[{self.name:s}] ", self.color) + line)

    def out(self, *args, **kwargs):
        if self.verbose > 0:
            line = self.__make_line(*args, **kwargs)
            sys.stdout.write(clr(f"[{self.name:s}] ", self.color) + line)

    # Writing to stdout (if verbose > 1)

    def rerr(self, *args, **kwargs):
        if self.verbose > 1:
            line = self.__make_line(*args, **kwargs)
            sys.stdout.write("\r\033[K" + clr(f"[{self.name:s}] ", self.color) +
                             line)

    def nerr(self, *args, **kwargs):
        if self.verbose > 1:
            line = self.__make_line(*args, **kwargs)
            sys.stdout.write("\n" + clr(f"[{self.name:s}] ", self.color) + line)

    def err(self, *args, **kwargs):
        if self.verbose > 1:
            line = self.__make_line(*args, **kwargs)
            sys.stdout.write(clr(f"[{self.name:s}] ", self.color) + line)
