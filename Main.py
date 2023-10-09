import Toplevels
import astra

import tkinter
from tkinter import filedialog
from tkinter.messagebox import showerror

import numpy as np
from numpy.linalg import norm

from Tabs import Tabs

from imageio import volread

from scipy.signal import convolve

FONT = {'10': ('Segoe UI', 10), '11': ('Segoe UI', 11)}

KW_ENTRIES = [{'font': FONT['10'], 'state': 'disabled'},
              {'column': 1, 'sticky': tkinter.SW, 'padx': (5, 10)}]

BEAM = {'Parallel': 'parallel3d_vec', 'Cone': 'cone_vec'}

ALGO = {'FBP': 'BP3D_CUDA', 'SIRT': 'SIRT3D_CUDA', 'CGLS': 'CGLS3D_CUDA',
        'DIRECTT': 'BP3D_CUDA'}


class Main(tkinter.Tk):
    def __init__(self):
        super().__init__()

        self.tabs = Tabs(self)

        tkinter.Button(self, width=12, text='Load data', font=FONT['11'],
                       command=self.load).grid(row=1, column=0, pady=5)

        self.reconstruct_button = tkinter.Button(self, width=12,
                                                 text='Reconstruct',
                                                 font=FONT['11'],
                                                 command=self.reconstruct)
        self.reconstruct_button.config(state='disabled')
        self.reconstruct_button.grid(row=1, column=1)
        self.save_button = tkinter.Button(self, width=12, text='Save volume',
                                          font=FONT['11'], command=self.save)
        self.save_button.config(state='disabled')
        self.save_button.grid(row=1, column=2)

    def load(self):
        filename = filedialog.askopenfilename(initialdir='C:/',
                                              title='Select file')

        self.title(filename.split('/')[-1])
        self.proj_data = volread(filename)
        self.proj_data[self.proj_data <= 0] = self.proj_data[
                self.proj_data > 0].min()
        self.data_shape = self.proj_data.shape

        if len(self.data_shape) == 2:
            self.proj_data = self.proj_data.reshape(self.data_shape[0], 1,
                                                    self.data_shape[1])
            self.data_shape = self.proj_data.shape

        self.proj_data = - np.log(self.proj_data).astype(np.float32).reshape(
                self.data_shape[0], self.data_shape[1] * self.data_shape[2])
        self.proj_range = np.append(np.amin(self.proj_data, axis=1).reshape(
                1, -1), np.amax(self.proj_data, axis=1).reshape(1, -1), axis=0)
        self.proj_data = self.proj_data.reshape(self.data_shape).transpose(
                1, 0, 2)

        proj = self.tabs.proj
        proj.proj_count.grid_forget()
        proj.proj_count = tkinter.Entry(proj, font=FONT['10'])
        proj.proj_count.insert(tkinter.END, str(self.data_shape[0]))
        proj.proj_count.config(state='readonly')
        proj.proj_count.grid(row=0, **KW_ENTRIES[1])
        proj.detector_row_count.grid_forget()
        proj.detector_row_count = tkinter.Entry(proj, font=FONT['10'])
        proj.detector_row_count.insert(tkinter.END, str(self.data_shape[2]))
        proj.detector_row_count.config(state='readonly')
        proj.detector_row_count.grid(row=1, **KW_ENTRIES[1])
        proj.detector_col_count.grid_forget()
        proj.detector_col_count = tkinter.Entry(proj, font=FONT['10'])
        proj.detector_col_count.insert(tkinter.END, str(self.data_shape[1]))
        proj.detector_col_count.config(state='readonly')
        proj.detector_col_count.grid(row=2, **KW_ENTRIES[1])
        proj.spacing_x.config(state='normal')
        proj.spacing_y.config(state='normal')

        geo = self.tabs.geo
        geo.start_angle.config(state='normal')
        geo.beam.config(state='normal')
        geo.offset_u.grid_forget()
        geo.offset_u = tkinter.Entry(geo, font=FONT['10'])
        geo.offset_u.insert(tkinter.END, str(self.data_shape[1]/2-.5))
        geo.offset_u.grid(row=4, **KW_ENTRIES[1])
        geo.offset_v.grid_forget()
        geo.offset_v = tkinter.Entry(geo, font=FONT['10'])
        geo.offset_v.insert(tkinter.END, str(self.data_shape[2]/2-.5))
        geo.offset_v.grid(row=5, **KW_ENTRIES[1])
        geo.scan_angle.config(state='normal')
        geo.direction.config(state='normal')
        geo.a.config(state='normal')
        geo.b.config(state='normal')
        geo.c.config(state='normal')

        vol = self.tabs.vol
        vol.grid_col_count.grid_forget()
        vol.grid_col_count = tkinter.Entry(vol, font=FONT['10'])
        vol.grid_col_count.insert(tkinter.END, str(self.data_shape[2]))
        vol.grid_col_count.grid(row=0, **KW_ENTRIES[1])
        vol.grid_row_count.grid_forget()
        vol.grid_row_count = tkinter.Entry(vol, font=FONT['10'])
        vol.grid_row_count.insert(tkinter.END, str(self.data_shape[2]))
        vol.grid_row_count.grid(row=1, **KW_ENTRIES[1])
        vol.grid_slice_count.grid_forget()
        vol.grid_slice_count = tkinter.Entry(vol, font=FONT['10'])
        vol.grid_slice_count.insert(tkinter.END, str(self.data_shape[1]))
        vol.grid_slice_count.grid(row=2, **KW_ENTRIES[1])
        vol.midpoint_x.config(state='normal')
        vol.midpoint_y.config(state='normal')
        vol.midpoint_z.config(state='normal')
        vol.voxel_x.config(state='normal')
        vol.voxel_y.config(state='normal')
        vol.voxel_z.config(state='normal')
        vol.output.config(state='normal')

        algo = self.tabs.algo
        algo.algo_menu.config(state='normal')
        algo.filters_menu.config(state='normal')

        self.reconstruct_button.config(state='normal')

        Toplevels.ProjImage(self)

    def reconstruct(self):
        proj = self.tabs.proj

        geo = self.tabs.geo

        vol = self.tabs.vol

        algo = self.tabs.algo

        pad_z = 0

        try:
            angles = np.linspace(np.radians(float(geo.start_angle.get())),
                                 np.radians(float(geo.start_angle.get())
                                 + float(geo.scan_angle.get())) * (1-2*(
                                         geo.direction_option.get(
                                                 ) == 'Clockwise')),
                                 self.data_shape[0], False)

            object_image_distance = float(geo.source_image_distance.get(
                    )) - float(geo.source_object_distance.get())

            vectors = np.zeros([self.data_shape[0], 12])
            vectors[:, 0] = np.sin(angles)
            vectors[:, 1] = - np.cos(angles)
            vectors[:, 5] = float(proj.spacing_y.get()) * (float(
                    geo.offset_v.get()) - int(proj.detector_row_count.get(
                            ))/2+.5)
            vectors[:, 6] = np.cos(angles) * float(proj.spacing_x.get(
                    )) * np.cos(np.radians(float(geo.b.get(
                            )))) * np.cos(np.radians(float(geo.c.get())))
            vectors[:, 7] = np.sin(angles) * float(proj.spacing_x.get(
                    )) * np.cos(np.radians(float(geo.b.get(
                            )))) * np.cos(np.radians(float(geo.c.get())))
            vectors[:, 8] = float(proj.spacing_x.get()) * np.sin(np.radians(
                    float(geo.b.get())))
            vectors[:, 9] = np.cos(angles) * float(proj.spacing_y.get(
                    )) * np.sin(np.radians(float(geo.b.get())))
            vectors[:, 10] = np.sin(angles) * float(proj.spacing_y.get(
                    )) * np.sin(np.radians(float(geo.b.get())))
            vectors[:, 11] = float(proj.spacing_y.get()) * np.cos(np.radians(
                    float(geo.a.get()))) * np.cos(np.radians(float(geo.b.get(
                            ))))

            if geo.beam_option.get() == 'Cone':
                vectors[:, :2] *= float(geo.source_object_distance.get())
                vectors[:, 3] = - np.sin(
                        angles) * object_image_distance + np.cos(
                                angles) * float(proj.spacing_x.get()) * (int(
                                      proj.detector_col_count.get(
                                              ))/2-.5-float(geo.offset_u.get(
                                                      )))
                vectors[:, 4] = np.cos(
                        angles) * object_image_distance + np.cos(
                                angles) * float(proj.spacing_x.get()) * (int(
                                        proj.detector_col_count.get(
                                                ))/2-.5-float(geo.offset_u.get(
                                                        )))

                if algo.algo_option.get() != 'FBP':
                    pad_z = int(vol.grid_slice_count.get()) * np.amax([
                            int(vol.grid_col_count.get()) * float(
                                    vol.voxel_x.get()), int(
                                            vol.grid_row_count.get()) * float(
                                                    vol.voxel_y.get())])
                    pad_z /= 2*float(geo.source_object_distance.get())
                    pad_z = int(pad_z - pad_z % 2)

                    k = int(algo.iterations.get())
                else:
                    k = 1

            self.vol_geom = astra.create_vol_geom(
                    int(vol.grid_row_count.get()),
                    int(vol.grid_col_count.get()),
                    int(vol.grid_slice_count.get()) + pad_z,
                    float(vol.midpoint_x.get()) - int(vol.grid_col_count.get(
                            ))/2 * float(vol.voxel_x.get()),
                    float(vol.midpoint_x.get()) + int(vol.grid_col_count.get(
                            ))/2 * float(vol.voxel_x.get()),
                    float(vol.midpoint_y.get()) - int(vol.grid_row_count.get(
                            ))/2 * float(vol.voxel_y.get()),
                    float(vol.midpoint_y.get()) + int(vol.grid_row_count.get(
                            ))/2 * float(vol.voxel_y.get()),
                    float(vol.midpoint_z.get()) - (int(
                            vol.grid_slice_count.get()) + pad_z)/2 * float(
                            vol.voxel_z.get()),
                    float(vol.midpoint_z.get()) + (int(
                            vol.grid_slice_count.get()) + pad_z)/2 * float(
                            vol.voxel_z.get()))
        except ValueError:
            showerror('Error', 'Invalid entry!')

        if algo.filter_option.get() == 'Ram-Lak':
            rampbl = np.zeros(self.data_shape[2]*2-self.data_shape[2] % 2)
            rampbl[self.data_shape[2]-1] = .25

            idxodd = np.concatenate((np.flip(- np.arange(1, rampbl.size//2, 2
                                                         ), axis=0), np.arange(
                            1, rampbl.size//2, 2)))

            rampbl[(self.data_shape[2] % 2)::2] = -1/(idxodd * np.pi)**2

            self.proj_data = convolve(self.proj_data, rampbl.reshape(1, 1, -1),
                                      mode='same')
        elif algo.filter_option.get() == 'Shepp-Logan':
            rampbl = -2/np.pi**2 / (4*np.arange(-self.data_shape[2]+1,
                                                self.data_shape[2] + (
                                                        self.data_shape[2]+1
                                                                ) % 2)**2-1)

            self.proj_data = convolve(self.proj_data, rampbl.reshape(1, 1, -1),
                                      mode='same')

        self.proj_geom = astra.create_proj_geom(BEAM[geo.beam_option.get(
                )], int(proj.detector_col_count.get()),
            int(proj.detector_row_count.get()), vectors)
        self.vol_id = astra.data3d.create('-vol', self.vol_geom)
        self.config = astra.astra_dict(ALGO[algo.algo_option.get()])
        self.config['ReconstructionDataId'] = self.vol_id

        if algo.algo_option.get() == 'DIRECTT':
            self.vol_data = self.directt(k)[pad_z//2:int(
                    vol.grid_slice_count.get())+pad_z//2]
        else:
            sino_id = astra.data3d.create('-sino', self.proj_geom,
                                          self.proj_data)

            self.config['ProjectionDataId'] = sino_id

            algo_id = astra.algorithm.create(self.config)

            astra.algorithm.run(algo_id, k)

            self.vol_data = astra.data3d.get(self.vol_id)[pad_z//2:int(
                    vol.grid_slice_count.get())+pad_z//2]

        self.vol_data = self.vol_data.reshape(int(vol.grid_slice_count.get()),
                                              int(vol.grid_row_count.get()) *
                                              int(vol.grid_col_count.get()))
        self.vol_range = np.append(np.amin(self.vol_data, axis=1).reshape(1,
                                   -1), np.amax(self.vol_data, axis=1).reshape(
                                           1, -1), axis=0)
        self.vol_data = self.vol_data.reshape(int(vol.grid_slice_count.get()),
                                              int(vol.grid_row_count.get()),
                                              int(vol.grid_col_count.get()))

        if algo.algo_option.get() != 'FBP':
            print(algo.algo_option.get(), 'was terminated after', k,
                  'iterations')

        self.save_button.config(state='normal')

        Toplevels.VolImage(self)

    def save(self):
        filename = self.tabs.vol.output.get()
        if filename == '':
            showerror('Error', 'No name was provided for the output file!')
        else:
            filename = filedialog.askdirectory(initialdir='C:/',
                                               title='Select directory'
                                               ) + '/' + filename
            if not filename.endswith('.raw'):
                filename += '.raw'
            np.uint8(255*(self.vol_data - self.vol_range[0].min())/(
                    self.vol_range[1].max() - self.vol_range[0].min())).tofile(
                filename)

    def directt(self, k):
        vol = self.tabs.vol

        res = np.copy(self.proj_data)

        norm_proj = norm(res)

        res_id = astra.data3d.create('-sino', self.proj_geom, res)

        uniform = np.ones([self.data_shape[1], self.data_shape[0],
                           self.data_shape[2]], dtype=np.float32)
        uniform *= norm_proj / norm(uniform)

        uni_id = astra.data3d.create('-sino', self.proj_geom, uniform)

        self.config['ProjectionDataId'] = uni_id

        algo_id = astra.algorithm.create(self.config)

        astra.algorithm.run(algo_id)

        backproj_uni = astra.data3d.get(self.vol_id)

        astra.data3d.delete(uni_id)
        astra.algorithm.delete(algo_id)

        uni_id, uniform = astra.create_sino3d_gpu(backproj_uni, self.proj_geom,
                                                  self.vol_geom)

        norm_uni = norm(uniform)

        del uniform

        vol_data = np.zeros([int(vol.grid_slice_count.get()), int(
                vol.grid_row_count.get()), int(vol.grid_col_count.get())],
            dtype=np.float32)

        for i in range(k):
            self.config['ProjectionDataId'] = res_id

            algo_id = astra.algorithm.create(self.config)

            astra.algorithm.run(algo_id)

            norm_res = norm(res)

            g_k = astra.data3d.get(
                    self.vol_id) - norm_res / norm_proj * backproj_uni
            g_k *= norm_res / norm_uni * backproj_uni / g_k.max()
            g_k[(g_k < 0) * (-g_k > vol_data)] = 0

            astra.data3d.delete(res_id)
            astra.algorithm.delete(algo_id)

            vol_data += g_k

            x_id, x_transform = astra.create_sino3d_gpu(vol_data,
                                                        self.proj_geom,
                                                        self.vol_geom)

            astra.data3d.delete(x_id)

            res = self.proj_data - x_transform

            res_id = astra.data3d.create('-sino', self.proj_geom, res)

        return(vol_data)


main = Main()
main.mainloop()
