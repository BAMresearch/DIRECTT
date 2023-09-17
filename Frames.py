import tkinter

FONT = {'10': ('Segoe UI', 10), '11': ('Segoe UI', 11)}

KW_LABELS = {'column': 0, 'sticky': tkinter.W, 'padx': (10, 5)}

KW_ENTRIES = [{'font': FONT['10'], 'state': 'disabled'},
              {'column': 1, 'sticky': tkinter.SW, 'padx': (5, 10)}]

BEAM = {'Parallel': 'parallel3d_vec', 'Cone': 'cone_vec'}

ALGO = {'FBP': 'BP3D_CUDA', 'SIRT': 'SIRT3D_CUDA', 'CGLS': 'CGLS3D_CUDA',
        'DIRECTT': 'BP3D_CUDA'}


class Proj(tkinter.Frame):
    def __init__(self, master):
        super().__init__(master, pady=10)

        self.grid(row=0, column=0)

        master.add(self, text=' Projections ')

        tkinter.Label(self, width=22, text='Projection count', anchor='w',
                      font=FONT['10']).grid(row=0, **KW_LABELS)
        tkinter.Label(self, text='Size X', font=FONT['10']).grid(
                row=1, **KW_LABELS)
        tkinter.Label(self, text='Size Y', font=FONT['10']).grid(
                row=2, **KW_LABELS)
        tkinter.Label(self, text='Projection pixel size X', font=FONT[
                '10']).grid(row=3, **KW_LABELS)
        tkinter.Label(self, text='Projection pixel size Y', font=FONT[
                '10']).grid(row=4, **KW_LABELS)

        self.proj_count = tkinter.Entry(self, **KW_ENTRIES[0])
        self.proj_count.grid(row=0, **KW_ENTRIES[1])
        self.detector_col_count = tkinter.Entry(self, **KW_ENTRIES[0])
        self.detector_col_count.grid(row=1, **KW_ENTRIES[1])
        self.detector_row_count = tkinter.Entry(self, **KW_ENTRIES[0])
        self.detector_row_count.grid(row=2, **KW_ENTRIES[1])
        self.spacing_x = tkinter.Entry(self, **KW_ENTRIES[0])
        self.spacing_x.grid(row=3, **KW_ENTRIES[1])
        self.spacing_y = tkinter.Entry(self, **KW_ENTRIES[0])
        self.spacing_y.grid(row=4, **KW_ENTRIES[1])


class Geo(tkinter.Frame):
    def __init__(self, master):
        super().__init__(master, pady=10)

        self.grid(row=0, column=1)

        master.add(self, text='  Geometry   ')

        tkinter.Label(self, width=22, text='Projection matrix start angle',
                      anchor='w', font=FONT['10']).grid(row=0, **KW_LABELS)
        tkinter.Label(self, text='Beam', font=FONT['10']).grid(
                row=1, **KW_LABELS)
        tkinter.Label(self, text='Source object distance', font=FONT[
                '10']).grid(row=2, **KW_LABELS)
        tkinter.Label(self, text='Source image distance', font=FONT[
                '10']).grid(row=3, **KW_LABELS)
        tkinter.Label(self, text='Detector offset u', font=FONT['10']).grid(
                row=4, **KW_LABELS)
        tkinter.Label(self, text='Detector offset v', font=FONT['10']).grid(
                row=5, **KW_LABELS)
        tkinter.Label(self, text='Scan angle', font=FONT['10']).grid(
                row=6, **KW_LABELS)
        tkinter.Label(self, text='Acquisition direction', font=FONT[
                '10']).grid(row=7, **KW_LABELS)
        tkinter.Label(self, text='a', font=FONT['10']).grid(row=8, **KW_LABELS)
        tkinter.Label(self, text='b', font=FONT['10']).grid(row=9, **KW_LABELS)
        tkinter.Label(self, text='c', font=FONT['10']).grid(
                row=10, **KW_LABELS)

        self.start_angle = tkinter.Entry(self)
        self.start_angle.insert(tkinter.END, '0')
        self.start_angle.config(**KW_ENTRIES[0])
        self.start_angle.grid(row=0, **KW_ENTRIES[1])
        self.beam_option = tkinter.StringVar()
        self.beam_option.set('Parallel')
        self.beam = tkinter.OptionMenu(self, self.beam_option, *list(
                BEAM.keys()), command=self.beam_command)
        self.beam.config(**KW_ENTRIES[0])
        self.beam.grid(row=1, **KW_ENTRIES[1])
        self.source_object_distance = tkinter.Entry(self, **KW_ENTRIES[0])
        self.source_object_distance.grid(row=2, **KW_ENTRIES[1])
        self.source_image_distance = tkinter.Entry(self, **KW_ENTRIES[0])
        self.source_image_distance.grid(row=3, **KW_ENTRIES[1])
        self.offset_u = tkinter.Entry(self, **KW_ENTRIES[0])
        self.offset_u.grid(row=4, **KW_ENTRIES[1])
        self.offset_v = tkinter.Entry(self, **KW_ENTRIES[0])
        self.offset_v.grid(row=5, **KW_ENTRIES[1])
        self.scan_angle = tkinter.Entry(self)
        self.scan_angle.insert(tkinter.END, '180')
        self.scan_angle.config(**KW_ENTRIES[0])
        self.scan_angle.grid(row=6, **KW_ENTRIES[1])
        self.direction_option = tkinter.StringVar()
        self.direction_option.set('Counter-clockwise')
        self.direction = tkinter.OptionMenu(self, self.direction_option,
                                            ['Clockwise', 'Counter-clockwise'])
        self.direction.config(**KW_ENTRIES[0])
        self.direction.grid(row=7, **KW_ENTRIES[1])
        self.a = tkinter.Entry(self)
        self.a.insert(tkinter.END, '0')
        self.a.config(**KW_ENTRIES[0])
        self.a.grid(row=8, **KW_ENTRIES[1])
        self.b = tkinter.Entry(self)
        self.b.insert(tkinter.END, '0')
        self.b.config(**KW_ENTRIES[0])
        self.b.grid(row=9, **KW_ENTRIES[1])
        self.c = tkinter.Entry(self)
        self.c.insert(tkinter.END, '0')
        self.c.config(**KW_ENTRIES[0])
        self.c.grid(row=10, **KW_ENTRIES[1])

    def beam_command(self, arg):
        self.scan_angle.grid_forget()

        if self.beam_option.get() == 'Parallel':
            self.source_object_distance.grid_forget()
            self.source_object_distance = tkinter.Entry(self)
            self.source_object_distance.config(**KW_ENTRIES[0])
            self.source_object_distance.grid(row=2, **KW_ENTRIES[1])
            self.source_image_distance.grid_forget()
            self.source_image_distance = tkinter.Entry(self)
            self.source_image_distance.config(**KW_ENTRIES[0])
            self.source_image_distance.grid(row=3, **KW_ENTRIES[1])
            self.scan_angle = tkinter.Entry(self, font=FONT['10'])
            self.scan_angle.insert(tkinter.END, '180')
            self.scan_angle.grid(row=6, **KW_ENTRIES[1])
        else:
            self.source_object_distance.config(state='normal')
            self.source_image_distance.config(state='normal')
            self.scan_angle = tkinter.Entry(self, font=FONT['10'])
            self.scan_angle.insert(tkinter.END, '360')
            self.scan_angle.grid(row=6, **KW_ENTRIES[1])


class Vol(tkinter.Frame):
    def __init__(self, master):
        super().__init__(master, pady=10)

        self.grid(row=0, column=2)

        master.add(self, text='   Volume    ')

        tkinter.Label(self, width=22, text='Size X', anchor='w', font=FONT[
                '10']).grid(row=0, **KW_LABELS)
        tkinter.Label(self, text='Size Y', font=FONT['10']).grid(
                row=1, **KW_LABELS)
        tkinter.Label(self, text='Size Z', font=FONT['10']).grid(
                row=2, **KW_LABELS)
        tkinter.Label(self, text='Midpoint X', font=FONT['10']).grid(
                row=3, **KW_LABELS)
        tkinter.Label(self, text='Midpoint Y', font=FONT['10']).grid(
                row=4, **KW_LABELS)
        tkinter.Label(self, text='Midpoint Z', font=FONT['10']).grid(
                row=5, **KW_LABELS)
        tkinter.Label(self, text='Voxel size X', font=FONT['10']).grid(
                row=6, **KW_LABELS)
        tkinter.Label(self, text='Voxel size Y', font=FONT['10']).grid(
                row=7, **KW_LABELS)
        tkinter.Label(self, text='Voxel size Z', font=FONT['10']).grid(
                row=8, **KW_LABELS)
        tkinter.Label(self, text='Volume output file', font=FONT['10']).grid(
                row=9, **KW_LABELS)

        self.grid_col_count = tkinter.Entry(self, **KW_ENTRIES[0])
        self.grid_col_count.grid(row=0, **KW_ENTRIES[1])
        self.grid_row_count = tkinter.Entry(self, **KW_ENTRIES[0])
        self.grid_row_count.grid(row=1, **KW_ENTRIES[1])
        self.grid_slice_count = tkinter.Entry(self, **KW_ENTRIES[0])
        self.grid_slice_count.grid(row=2, **KW_ENTRIES[1])
        self.midpoint_x = tkinter.Entry(self)
        self.midpoint_x.insert(tkinter.END, '0')
        self.midpoint_x.config(**KW_ENTRIES[0])
        self.midpoint_x.grid(row=3, **KW_ENTRIES[1])
        self.midpoint_y = tkinter.Entry(self)
        self.midpoint_y.insert(tkinter.END, '0')
        self.midpoint_y.config(**KW_ENTRIES[0])
        self.midpoint_y.grid(row=4, **KW_ENTRIES[1])
        self.midpoint_z = tkinter.Entry(self)
        self.midpoint_z.insert(tkinter.END, '0')
        self.midpoint_z.config(**KW_ENTRIES[0])
        self.midpoint_z.grid(row=5, **KW_ENTRIES[1])
        self.voxel_x = tkinter.Entry(self, **KW_ENTRIES[0])
        self.voxel_x.grid(row=6, **KW_ENTRIES[1])
        self.voxel_y = tkinter.Entry(self, **KW_ENTRIES[0])
        self.voxel_y.grid(row=7, **KW_ENTRIES[1])
        self.voxel_z = tkinter.Entry(self, **KW_ENTRIES[0])
        self.voxel_z.grid(row=8, **KW_ENTRIES[1])
        self.output = tkinter.Entry(self)
        self.output.insert(tkinter.END, 'volume.raw')
        self.output.config(**KW_ENTRIES[0])
        self.output.grid(row=9, **KW_ENTRIES[1])


class Algo(tkinter.Frame):
    def __init__(self, master):
        super().__init__(master, pady=10)

        self.grid(row=0, column=3)

        master.add(self, text='  Algorithm  ')

        tkinter.Label(self, width=22, text='Algorithm', anchor='w', font=FONT[
                '10']).grid(row=0, **KW_LABELS)
        tkinter.Label(self, text='Filter', font=FONT['10']).grid(
                row=1, **KW_LABELS)
        tkinter.Label(self, text='Iterations', font=FONT['10']).grid(
                row=2, **KW_LABELS)

        self.algo_option = tkinter.StringVar()
        self.algo_option.set('FBP')
        self.algo_menu = tkinter.OptionMenu(self, self.algo_option, *list(
                ALGO.keys()), command=self.algo_command)
        self.algo_menu.config(**KW_ENTRIES[0])
        self.algo_menu.grid(row=0, **KW_ENTRIES[1])
        self.filter_option = tkinter.StringVar()
        self.filter_option.set('Ram-Lak')
        self.filters_menu = tkinter.OptionMenu(self, self.filter_option,
                                               *['Ram-Lak', 'Shepp-Logan',
                                                 'None'])
        self.filters_menu.config(**KW_ENTRIES[0])
        self.filters_menu.grid(row=1, **KW_ENTRIES[1])
        self.iterations = tkinter.Entry(self, **KW_ENTRIES[0])
        self.iterations.grid(row=2, **KW_ENTRIES[1])

    def algo_command(self, arg):
        if self.algo_option.get() == 'FBP':
            self.filter_option.set('Ram-Lak')
            self.filters_menu.config(state='normal')
            self.iterations.config(state='disabled')
        elif self.algo_option.get() == 'DIRECTT':
            self.filter_option.set('None')
            self.filters_menu.config(state='disabled')
            self.iterations.config(state='normal')
        else:
            self.filter_option.set('None')
            self.filters_menu.config(state='disabled')
            self.iterations.config(state='normal')
