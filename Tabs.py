import Frames

from tkinter.ttk import Notebook


class Tabs(Notebook):
    def __init__(self, master):
        super().__init__(master)

        self.grid(row=0, column=0, columnspan=3)
        self.proj = Frames.Proj(self)
        self.geo = Frames.Geo(self)
        self.vol = Frames.Vol(self)
        self.algo = Frames.Algo(self)
