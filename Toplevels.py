import tkinter

from numpy import uint8

from PIL.Image import fromarray
from PIL.ImageTk import PhotoImage


class ProjImage(tkinter.Toplevel):
    def __init__(self, master):
        super().__init__()

        self.proj_data = master.proj_data
        self.proj_range = master.proj_range

        if master.data_shape[1] > master.data_shape[2]:
            self.proj_height = 1152
            self.proj_width = 1151*master.data_shape[2] // master.data_shape[
                    1]+1
        else:
            self.proj_width = 1152
            self.proj_height = 1151*master.data_shape[1] // master.data_shape[
                    2]+1

        self.proj_var = tkinter.IntVar()
        self.proj_var.set(master.data_shape[0]//2+1)

        view = self.proj_var.get()-1

        image = fromarray(uint8(255*(self.proj_data[:, view] - self.proj_range[
                0, view]) / (self.proj_range[1, view] - self.proj_range[
                        0, view])))

        self.proj_photo = PhotoImage(image=image.resize((self.proj_width,
                                                         self.proj_height)))
        self.proj_image = tkinter.Label(self, image=self.proj_photo)
        self.proj_image.grid(row=0, pady=15)

        if master.data_shape[0] > 38:
            sliderlength = 30
        else:
            sliderlength = int(1152/master.data_shape[0])

        proj_scale = tkinter.Scale(self, variable=self.proj_var, from_=1,
                                   to=master.data_shape[0],
                                   orient=tkinter.HORIZONTAL, length=1152,
                                   sliderlength=sliderlength,
                                   command=self.proj_scale_command)
        proj_scale.grid(row=1)

    def proj_scale_command(self, args):
        view = self.proj_var.get()-1

        image = fromarray(uint8(255*(self.proj_data[:, view] - self.proj_range[
                0, view]) / (self.proj_range[1, view] - self.proj_range[
                        0, view])))

        self.proj_image.grid_forget()
        self.proj_photo = PhotoImage(image=image.resize((self.proj_width,
                                                         self.proj_height)))
        self.proj_image = tkinter.Label(self, image=self.proj_photo)
        self.proj_image.grid(row=0, pady=15)


class VolImage(tkinter.Toplevel):
    def __init__(self, master):
        super().__init__()

        vol = master.tabs.vol

        self.vol_data = master.vol_data
        self.vol_range = master.vol_range

        if int(vol.grid_col_count.get()) > int(vol.grid_row_count.get()):
            self.vol_width = 1152
            self.vol_height = int(1151*int(vol.grid_row_count.get()) / int(
                    vol.grid_col_count.get()))+1
        else:
            self.vol_height = 1152
            self.vol_width = int(1151*int(vol.grid_col_count.get()) / int(
                    vol.grid_row_count.get()))+1

        self.vol_var = tkinter.IntVar()
        self.vol_var.set(int(vol.grid_slice_count.get())//2+1)

        view = self.vol_var.get()-1

        self.vol_photo = PhotoImage(image=fromarray(uint8(255*(
                self.vol_data[view] - self.vol_range[0, view]) / (
                        self.vol_range[1, view] - self.vol_range[0, view]
                        ))).resize((self.vol_width, self.vol_height)))
        self.vol_image = tkinter.Label(self, image=self.vol_photo)
        self.vol_image.grid(row=0, pady=15)

        if int(vol.grid_slice_count.get()) > 38:
            sliderlength = 30
        else:
            sliderlength = int(1152/int(vol.grid_slice_count.get()))

        vol_scale = tkinter.Scale(self, variable=self.vol_var, from_=1,
                                  to=vol.grid_slice_count.get(),
                                  orient=tkinter.HORIZONTAL, length=1152,
                                  sliderlength=sliderlength,
                                  command=self.vol_scale_command)
        vol_scale.grid(row=1)

    def vol_scale_command(self, args):
        view = self.vol_var.get()-1

        self.vol_image.grid_forget()
        self.vol_photo = PhotoImage(image=fromarray(uint8(255*(self.vol_data[
                view] - self.vol_range[0, view]) / (self.vol_range[
                        1, view] - self.vol_range[0, view]))).resize(
                (self.vol_width, self.vol_height)))
        self.vol_image = tkinter.Label(self, image=self.vol_photo)
        self.vol_image.grid(row=0, pady=15)
