import tkinter
import tkinter.filedialog
import tkinter.ttk
import tkinter.messagebox
import numpy as np
import scipy.ndimage
import scipy.signal
import imageio
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageTk
import astra
import astra.utils


def load():
    
    global transmission, NumRows, NumProjections, NumChannelsPerRow
    global count, ProjSizeX, ProjSizeY
    global offsetU, offsetV
    global sizeX, sizeY, sizeZ
    global projHeight, projWidth, proj_img, projections, projSlider
    
    filename = tkinter.filedialog.askopenfilename(
        initialdir = 'C:/', title = 'Select a file' )
    
    root.title( filename )
    
    transmission = imageio.volread( filename ).transpose( 1, 0, 2 )
    
    NumRows, NumProjections, NumChannelsPerRow = transmission.shape
    
    count.grid_forget()
    
    ProjSizeX.grid_forget()
    ProjSizeY.grid_forget()
    
    offsetU.grid_forget()
    offsetV.grid_forget()
    
    sizeX.grid_forget()
    sizeY.grid_forget()
    sizeZ.grid_forget()
    
    count = tkinter.Entry( projPanel )
    count.insert( tkinter.END, str( NumProjections ) )
    count.config( state = 'readonly' )
    count.grid( row = 0, column = 1 )
    
    ProjSizeX = tkinter.Entry( projPanel )
    ProjSizeX.insert( tkinter.END, str( NumChannelsPerRow ) )
    ProjSizeX.config( state = 'readonly' )
    ProjSizeX.grid( row = 1, column = 1 )
    
    ProjSizeY = tkinter.Entry( projPanel )
    ProjSizeY.insert( tkinter.END, str( NumRows ) )
    ProjSizeY.config( state = 'readonly' )
    ProjSizeY.grid( row = 2, column = 1 )
    
    PixelSizeU.config( state = 'normal' )
    PixelSizeV.config( state = 'normal' )
    
    start.config( state = 'normal' )
    
    beam.config( state = 'normal' )
    
    offsetU = tkinter.Entry( geoPanel )
    offsetU.insert( tkinter.END, str( NumChannelsPerRow / 2 - .5 ) )
    offsetU.grid( row = 4, column = 1, sticky = tkinter.E )
    
    offsetV = tkinter.Entry( geoPanel )
    offsetV.insert( tkinter.END, str( NumRows / 2 - .5 ) )
    offsetV.grid( row = 5, column = 1, sticky = tkinter.E )
    
    scanAngle.config( state = 'normal' )
    
    direction.config( state = 'normal' )
    
    a.config( state = 'normal' )
    b.config( state = 'normal' )
    c.config( state = 'normal' )
    
    sizeX = tkinter.Entry( volPanel )
    sizeX.insert( tkinter.END, str( NumChannelsPerRow ) )
    sizeX.grid( row = 0, column = 1 )
    
    sizeY = tkinter.Entry( volPanel )
    sizeY.insert( tkinter.END, str( NumChannelsPerRow ) )
    sizeY.grid( row = 1, column = 1 )
    
    sizeZ = tkinter.Entry( volPanel )
    sizeZ.insert( tkinter.END, str( NumRows ) )
    sizeZ.grid( row = 2, column = 1 )
    
    midpointX.config( state = 'normal' )
    midpointY.config( state = 'normal' )
    midpointZ.config( state = 'normal' )
    
    voxelSizeX.config( state = 'normal' )
    voxelSizeY.config( state = 'normal' )
    voxelSizeZ.config( state = 'normal' )
    
    output.config( state = 'normal' )
    
    algoMenu.config( state = 'normal' )
    
    filterMenu.config( state = 'normal' )
    
    if NumRows > NumChannelsPerRow:
        
        projHeight = 610
        
        projWidth = np.ceil(
            610 * NumChannelsPerRow / NumRows ).astype( np.int )
    
    else:
        
        projWidth = 610
        
        projHeight = np.ceil(
            610 * NumRows / NumChannelsPerRow ).astype( np.int )
    
    proj_img = PIL.ImageTk.PhotoImage( image = PIL.Image.fromarray(
        np.uint8( ( transmission[ :, NumProjections // 2, : ] - np.amin(
        transmission ) ) / ( np.amax( transmission ) - np.amin(
        transmission ) ) * 255 ) ).resize( ( projWidth, projHeight ) ) )
    
    projections = tkinter.Label( projSlices, image = proj_img )
    projections.grid( row = 0 )
    
    projVar = tkinter.IntVar()
    projVar.set( NumProjections // 2 + 1 )
    
    projSlider = tkinter.Scale( projSlices, variable = projVar, from_ = 1,
        to = NumProjections,
        orient = tkinter.HORIZONTAL, length = 610, command = pSlider )
    projSlider.grid( row = 1 )
    
    recButton.config( state = 'normal' )

def beamF( var ):
    
    global scanAngle, cone
    
    scanAngle.grid_forget()
    
    if var == 'parallel':
        
        SOD.config( state = 'disabled' )
        
        SDD.config( state = 'disabled' )
        
        scanAngle = tkinter.Entry( geoPanel )
        scanAngle.insert( tkinter.END, '180' )
        scanAngle.grid( row = 6, column = 1, sticky = tkinter.E )
        
        cone = False
    
    else:
        
        SOD.config( state = 'normal' )
        
        SDD.config( state = 'normal' )
        
        scanAngle = tkinter.Entry( geoPanel )
        scanAngle.insert( tkinter.END, '360' )
        scanAngle.grid( row = 6, column = 1, sticky = tkinter.E )
        
        cone = True

def dirF( var ):
    
    global counter
    
    counter = var == 'counter-clockwise'

def algoF( var ):
    
    global filterMenu, iterations, window, algorithm
    
    filterMenu.grid_forget()
    
    iterations.grid_forget()
    
    if var == algoList[ 0 ]:
        
        filterVar.set( filterList[ 0 ] )
        
        filterMenu = tkinter.OptionMenu(
            algoPanel, filterVar, *filterList, command = windowF )
        filterMenu.grid( row = 1, column = 1, sticky = tkinter.W )
        
        iterations = tkinter.Entry( algoPanel )
        iterations.insert( tkinter.END, '1' )
        iterations.config( state = 'disabled' )
        iterations.grid( row = 2, column = 1 )
    
    elif var in algoList[ 1:3 ]:
        
        filterVar.set( filterList[ -1 ] )
        
        filterMenu = tkinter.OptionMenu( algoPanel, filterVar, *filterList )
        filterMenu.config( state = 'disabled' )
        filterMenu.grid( row = 1, column = 1, sticky = tkinter.W )
        
        window = 'none'
        
        iterations = tkinter.Entry( algoPanel )
        iterations.grid( row = 2, column = 1 )
    
    else:
        
        filterVar.set( filterList[ -1 ] )
        
        filterMenu = tkinter.OptionMenu( algoPanel, filterVar, *filterList )
        filterMenu.config( state = 'disabled' )
        filterMenu.grid( row = 1, column = 1, sticky = tkinter.W )
        
        window = 'none'
        
        iterations = tkinter.Entry( algoPanel )
        iterations.insert( tkinter.END, '1' )
        iterations.config( state = 'disabled' )
        iterations.grid( row = 2, column = 1 )
    
    algorithm = var

def windowF( var ):
    
    global window
    
    window = var

def pSlider( var ):
    
    global projections, proj_img
    
    projections.grid_forget()
    
    proj_img = PIL.ImageTk.PhotoImage( image = PIL.Image.fromarray(
        np.uint8( ( transmission[ :, int( var ) - 1, : ] - np.amin(
        transmission ) ) / ( np.amax( transmission ) - np.amin(
        transmission ) ) * 255 ) ).resize( ( projWidth, projHeight ) ) )
    
    projections = tkinter.Label( projSlices, image = proj_img )
    projections.grid( row = 0 )

def rSlider( var ):
    
    global volume, rec_img
    
    volume.grid_forget()
    
    rec_img = PIL.ImageTk.PhotoImage( image = PIL.ImageEnhance.Contrast(
        PIL.Image.fromarray( np.uint8( ( rec[ int( var ) ] - np.amin(
        rec ) ) / ( np.amax( rec ) - np.amin(
        rec ) ) * 255 ) ) ).enhance( 2 ).resize( ( recWidth, recHeight ) ) )
        
    volume = tkinter.Label( volSlices, image = rec_img )
    volume.grid( row = 0 )

def reconstruct():
    
    global NumSlices, rec, recWidth, recHeight, volume, rec_img, recSlider
    
    try:
        
        proj = - np.log( transmission )
        
        theta = np.linspace( np.radians( float( start.get() ) ), np.radians(
            float( start.get() ) + float( scanAngle.get() ) ) * counterDict[
            counter ], NumProjections, False )
        
        vectors = np.zeros( [ NumProjections, 12 ] )
        vectors[ :, 0 ] = np.sin( theta ) * float( SOD.get() )
        vectors[ :, 1 ] = - np.cos( theta ) * float( SOD.get() )
        vectors[ :, 3 ] = - np.sin( theta ) * ( float( SDD.get() ) - float(
            SOD.get() ) ) + np.cos( theta ) * ( (
            NumChannelsPerRow / 2 - .5 ) - float(
            offsetU.get() ) ) * float( PixelSizeU.get() )
        vectors[ :, 4 ] = np.cos( theta ) * ( float( SDD.get() ) - float(
            SOD.get() ) ) + np.sin( theta ) * ( (
            NumChannelsPerRow / 2 - .5 ) - float(
            offsetU.get() ) ) * float( PixelSizeU.get() )
        vectors[ :, 5 ] = ( ( NumRows / 2 - .5 ) - float(
            offsetV.get() )  ) * float( PixelSizeV.get() )
        vectors[ :, 6 ] = np.cos( theta ) * np.cos( np.radians( float(
            b.get() ) ) ) * np.cos(
            np.radians( float( c.get() ) ) ) * float( PixelSizeU.get() )
        vectors[ :, 7 ] = np.sin( theta ) * np.cos( np.radians( float(
            b.get() ) ) ) * np.cos(
            np.radians( float( c.get() ) ) ) * float( PixelSizeU.get() )
        vectors[ :, 8 ] = np.sin(
            np.radians( float( b.get() ) ) ) * float( PixelSizeU.get() )
        vectors[ :, 9 ] = np.cos( theta ) * np.sin(
            np.radians( float( b.get() ) ) ) * float( PixelSizeV.get() )
        vectors[ :, 10 ] = np.sin( theta ) * np.sin(
            np.radians( float( b.get() ) ) ) * float( PixelSizeV.get() )
        vectors[ :, 11 ] = np.cos( np.radians( float( a.get() ) ) ) * np.cos(
            np.radians( float( b.get() ) ) ) * float( PixelSizeV.get() )
        
        NumSlices = int( int( sizeZ.get() ) * (
            1 + NumChannelsPerRow * float( voxelSizeZ.get() ) / (
            2 * float( SOD.get() ) ) * ( algorithm != 'FBP' ) * cone ) )
        
        if NumSlices % 2 != int( sizeZ.get() ) % 2:
            
            NumSlices -= 1
        
        vol_geom = astra.create_vol_geom( int( sizeY.get() ), int(
            sizeX.get() ), NumSlices, float( midpointX.get() ) - float(
            voxelSizeX.get() ) * int( sizeX.get() ) / 2, float(
            midpointX.get() ) + float( voxelSizeX.get() ) * int(
            sizeX.get() ) / 2, float( midpointY.get() ) - float(
            voxelSizeY.get() ) * int( sizeY.get() ) / 2, float(
            midpointY.get() ) + float( voxelSizeY.get() ) * int(
            sizeY.get() ) / 2, float( midpointZ.get() ) - float(
            voxelSizeZ.get() ) * NumSlices / 2, float(
            midpointZ.get() ) + float( voxelSizeZ.get() ) * NumSlices / 2 )
        
        proj_geom = astra.create_proj_geom(
            geomDict[ cone ], NumRows, NumChannelsPerRow, vectors )
        
        if algorithm == 'DIRECTT':
            
            sumProj = np.sum( proj[ int( float( offsetV.get() ) ) ] )
            
            residual = np.copy( proj )
            
            rec = np.zeros( [
                NumSlices, int( sizeY.get() ), int( sizeX.get() ) ] )
            
            if cone:
                
                C = np.linspace( - NumChannelsPerRow /
                    2, NumChannelsPerRow / 2, NumChannelsPerRow )
                
                x, y = np.meshgrid( C, C )
                
                cylinder = np.ones( rec.shape, dtype = np.int ) * np.array(
                    x ** 2 + y ** 2 < ( NumChannelsPerRow / 2 ) ** 2 ).reshape(
                    1, NumChannelsPerRow, NumChannelsPerRow )
                
                fp_id, fp = astra.create_sino3d_gpu(
                    cylinder, proj_geom, vol_geom )
                fp /= fp[ int( float( offsetV.get() )
                ) ].reshape( 1, NumProjections, NumChannelsPerRow )
                
                bp_id = astra.data3d.create( '-vol', vol_geom )
                
                proj_id = astra.data3d.create( '-sino', proj_geom, proj )
                
                cfg = astra.astra_dict( 'BP3D_CUDA' )
                cfg[ 'ReconstructionDataId' ] = bp_id
                cfg[ 'ProjectionDataId' ] = proj_id
                
                alg_id = astra.algorithm.create( cfg )
                
                astra.algorithm.run( alg_id )
                
                model = astra.data3d.get( bp_id )
                model /= model[ NumSlices // 2 ].reshape(
                    1, int( sizeY.get() ), int( sizeX.get() ) )
            
            first = True
            
            while 1:
                
                bp_id = astra.data3d.create( '-vol', vol_geom )
                
                proj_id = astra.data3d.create( '-sino', proj_geom, residual )
                
                cfg = astra.astra_dict( 'BP3D_CUDA' )
                cfg[ 'ReconstructionDataId' ] = bp_id
                cfg[ 'ProjectionDataId' ] = proj_id
                
                alg_id = astra.algorithm.create( cfg )
                
                astra.algorithm.run( alg_id )
                
                bp = astra.data3d.get( bp_id )
                
                if cone:
                    
                    bp[ model > 0 ] /= model[ model > 0 ]
                
                if first:
                    
                    mass = np.rint( scipy.ndimage.center_of_mass(
                        bp[ NumSlices // 2 ] ) ).astype( np.int )
                    
                    selection = np.amax( bp[ NumSlices // 2 ] ) - bp[
                        NumSlices // 2, mass[ 0 ], mass[ 1 ] ]
                    
                    first = False
                
                threshold = np.amax( bp[ NumSlices // 2 ][ bp[
                    NumSlices // 2 ] < np.amax( bp[
                    NumSlices // 2 ] ) - np.sum( residual[
                    int( float( offsetV.get() ) ) ] ) / sumProj * selection ] )
                
                bp -= threshold
                bp *= bp > 0
                
                rec += bp
                
                fp_id, fp = astra.create_sino3d_gpu(
                    rec, proj_geom, vol_geom )
                
                residual = proj - fp
                
                if np.any( np.sum( residual.transpose( 1, 0, 2 ).reshape(
                    NumProjections,
                    NumRows * NumChannelsPerRow ), axis = 1 ) <= 0 ):
                    
                    break
                
                astra.functions.clear()
            
        else:
            
            bp_id = astra.data3d.create( '-vol', vol_geom )
            
            proj_id = astra.data3d.create(
                '-sino', proj_geom, filterF( proj, NumChannelsPerRow ) )
            
            cfg = astra.astra_dict( algoDict[ algorithm ] )
            cfg[ 'ReconstructionDataId' ] = bp_id
            cfg[ 'ProjectionDataId' ] = proj_id
            
            alg_id = astra.algorithm.create( cfg )
            
            astra.algorithm.run( alg_id, int( iterations.get() ) )
            
            rec = astra.data3d.get( bp_id )
            
            if algorithm == 'FBP':
                
                rec /= ( float( voxelSizeX.get() ) * float(
                    voxelSizeY.get() ) * float( voxelSizeZ.get() ) ) ** (
                1 / 3 ) * np.abs( theta[ 1 ] - theta[ 0 ] )
        
        if NumSlices > int( sizeZ.get() ):
            
            rec = rec[ ( ( NumSlices - int( sizeZ.get() ) ) // 2 ):(
                - ( NumSlices - int( sizeZ.get() ) ) // 2 ) ]
        
        slices.add( volSlices, text = 'Volume', state = 'normal' )
        
        if int( sizeX.get() ) > int( sizeY.get() ):
            
            recWidth = 610
            
            recHeight = np.ceil(
                610 * int( sizeY.get() ) / int( sizeX.get() ) ).astype( int )
        
        else:
            
            recHeight = 610
            
            recWidth = np.ceil(
                610 * int( sizeX.get() ) / int( sizeY.get() ) ).astype( int )
        
        rec_img = PIL.ImageTk.PhotoImage( image = PIL.ImageEnhance.Contrast(
            PIL.Image.fromarray( np.uint8( ( rec[ int(
            sizeZ.get() ) // 2 ] - np.amin( rec ) ) / ( np.amax(
            rec ) - np.amin( rec ) ) * 255 ) ) ).enhance(
            2 ).resize( ( recWidth, recHeight ) ) )
        
        volume = tkinter.Label( volSlices, image = rec_img )
        volume.grid( row = 0 )
        
        recVar = tkinter.IntVar()
        recVar.set( int( sizeZ.get() ) // 2 + 1 )
        
        recSlider = tkinter.Scale(
            volSlices, variable = recVar, from_ = 1, to = int( sizeZ.get() ),
            orient = tkinter.HORIZONTAL, length = 610, command = rSlider )
        recSlider.grid( row = 1 )
        
        saveButton.config( state = 'normal' )
    
    except ValueError:
        
        tkinter.messagebox.showerror( 'Error', 'Invalid entry!' )

def filterF( sino, s ):
    
    if window == 'Ram-Lak':
        
        rampbl = np.zeros( s * 2 - s % 2, dtype = np.float32 )    
        rampbl[ s - 1 ] = .25
        
        idxodd = np.concatenate( ( np.flip( -1 * np.arange(
            1, rampbl.size // 2, 2 ), axis = 0 ), np.arange(
            1, rampbl.size // 2, 2 ) ) )
        
        rampbl[ ( s % 2 )::2 ] = -1 / ( idxodd * np.pi ) ** 2
        
        return scipy.signal.convolve(
            sino, rampbl.reshape( [ 1, 1, rampbl.size ] ), mode = 'same' )
    
    elif window == 'Shepp-Logan':
        
        rampbl = -2 / np.pi ** 2 / (
            4 * np.arange( - s + 1, s + ( s + 1 ) % 2 ) ** 2 - 1 )
        
        return scipy.signal.convolve(
            sino, rampbl.reshape( [ 1, 1, rampbl.size ] ), mode = 'same' )
    
    else:
        
        return sino

def save():
    
    directory = tkinter.filedialog.askdirectory(
        initialdir = 'C:/', title = 'Select a folder' )
    
    rec.tofile( directory + '/' + output.get() )

root = tkinter.Tk()

parameters = tkinter.ttk.Notebook( root, height = 655, width = 306 )
parameters.grid( row = 0, column = 0, columnspan = 2, padx = 5, pady = 5 )

projPanel = tkinter.Frame( parameters )
projPanel.grid( row = 0, column = 0 )

parameters.add( projPanel, text = 'Projections' )

tkinter.Label( projPanel, text = 'Projection count (circular scan)' ).grid(
    row = 0, column = 0, sticky = tkinter.W )
tkinter.Label( projPanel, text = 'Size X' ).grid(
    row = 1, column = 0, sticky = tkinter.W )
tkinter.Label( projPanel, text = 'Size Y' ).grid(
    row = 2, column = 0, sticky = tkinter.W )
tkinter.Label( projPanel, text = 'Projection pixel size X' ).grid(
    row = 3, column = 0, sticky = tkinter.W )
tkinter.Label( projPanel, text = 'Projection pixel size Y' ).grid(
    row = 4, column = 0, sticky = tkinter.W )

count = tkinter.Entry( projPanel, state = 'disabled' )
count.grid( row = 0, column = 1, sticky = tkinter.E )

ProjSizeX = tkinter.Entry( projPanel, state = 'disabled' )
ProjSizeX.grid( row = 1, column = 1 )

ProjSizeY = tkinter.Entry( projPanel, state = 'disabled' )
ProjSizeY.grid( row = 2, column = 1 )

PixelSizeU = tkinter.Entry( projPanel, state = 'disabled' )
PixelSizeU.grid( row = 3, column = 1 )

PixelSizeV = tkinter.Entry( projPanel, state = 'disabled' )
PixelSizeV.grid( row = 4, column = 1 )

geoPanel = tkinter.Frame( parameters )
geoPanel.grid( row = 0, column = 1 )

parameters.add( geoPanel, text = 'Geometry' )

tkinter.Label( geoPanel, text = 'Projection matrix start angle' ).grid(
    row = 0, column = 0, sticky = tkinter.W )
tkinter.Label(
    geoPanel, text = 'Beam' ).grid( row = 1, column = 0, sticky = tkinter.W )
tkinter.Label( geoPanel, text = 'Source object distance' ).grid(
    row = 2, column = 0, sticky = tkinter.W )
tkinter.Label( geoPanel, text = 'Source image distance' ).grid(
    row = 3, column = 0, sticky = tkinter.W )
tkinter.Label( geoPanel, text = 'Detector offset u' ).grid(
    row = 4, column = 0, sticky = tkinter.W )
tkinter.Label( geoPanel, text = 'Detector offset v' ).grid(
    row = 5, column = 0, sticky = tkinter.W )
tkinter.Label( geoPanel, text = 'Scan angle' ).grid(
    row = 6, column = 0, sticky = tkinter.W )
tkinter.Label( geoPanel, text = 'Acquisition direction' ).grid(
    row = 7, column = 0, sticky = tkinter.W )
tkinter.Label(
    geoPanel, text = 'a' ).grid( row = 8, column = 0, sticky = tkinter.W )
tkinter.Label(
    geoPanel, text = 'b' ).grid( row = 9, column = 0, sticky = tkinter.W  )
tkinter.Label(
    geoPanel, text = 'c' ).grid( row = 10, column = 0, sticky = tkinter.W )

start = tkinter.Entry( geoPanel )
start.insert( tkinter.END, '0' )
start.config( state = 'disabled' )
start.grid( row = 0, column = 1, sticky = tkinter.E )

beamOptions = [
    'parallel',
    'cone' ]

beamVar = tkinter.StringVar( geoPanel )
beamVar.set( beamOptions[ 0 ] )

cone = False

geomDict = {
    False: 'parallel3d_vec',
    True: 'cone_vec' }

beam = tkinter.OptionMenu(
    geoPanel, beamVar, *beamOptions, command = beamF )
beam.config( state = 'disabled' )
beam.grid( row = 1, column = 1, sticky = tkinter.E )

SOD = tkinter.Entry( geoPanel )
SOD.insert( tkinter.END, '1' )
SOD.config( state = 'disabled' )
SOD.grid( row = 2, column = 1, sticky = tkinter.E )

SDD = tkinter.Entry( geoPanel )
SDD.insert( tkinter.END, '2' )
SDD.config( state = 'disabled' )
SDD.grid( row = 3, column = 1, sticky = tkinter.E )

offsetU = tkinter.Entry( geoPanel, state = 'disabled' )
offsetU.grid( row = 4, column = 1, sticky = tkinter.E )

offsetV = tkinter.Entry( geoPanel, state = 'disabled' )
offsetV.grid( row = 5, column = 1, sticky = tkinter.E )

scanAngle = tkinter.Entry( geoPanel )
scanAngle.insert( tkinter.END, '180' )
scanAngle.config( state = 'disabled' )
scanAngle.grid( row = 6, column = 1, sticky = tkinter.E )

dirOptions = [
    'clockwise',
    'counter-clockwise' ]

dirVar = tkinter.StringVar()
dirVar.set( dirOptions[ 1 ] )

counter = True

counterDict = {
    False: -1,
    True: 1 }

direction = tkinter.OptionMenu(
        geoPanel, dirVar, *dirOptions, command = dirF )
direction.config( width = 17, state = 'disabled' )
direction.grid( row = 7, column = 1, sticky = tkinter.E )

a = tkinter.Entry( geoPanel )
a.insert( tkinter.END, '0' )
a.config( state = 'disabled' )
a.grid( row = 8, column = 1, sticky = tkinter.E )

b = tkinter.Entry( geoPanel )
b.insert( tkinter.END, '0' )
b.config( state = 'disabled' )
b.grid( row = 9, column = 1, sticky = tkinter.E )

c = tkinter.Entry( geoPanel )
c.insert( tkinter.END, '0' )
c.config( state = 'disabled' )
c.grid( row = 10, column = 1, sticky = tkinter.E )

volPanel = tkinter.Frame( parameters )
volPanel.grid( row = 0, column = 2 )

parameters.add( volPanel, text = 'Volume' )

tkinter.Label( volPanel, text = 'Size X' ).grid(
    row = 0, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Size Y' ).grid(
    row = 1, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Size Z' ).grid(
    row = 2, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Midpoint X' ).grid(
    row = 3, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Midpoint Y' ).grid(
    row = 4, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Midpoint Z' ).grid(
    row = 5, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Voxel size X' ).grid(
    row = 6, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Voxel size Y' ).grid(
    row = 7, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Voxel size Z' ).grid(
    row = 8, column = 0, sticky = tkinter.W )
tkinter.Label( volPanel, text = 'Volume output file' ).grid(
    row = 9, column = 0, sticky = tkinter.W )

sizeX = tkinter.Entry( volPanel, state = 'disabled' )
sizeX.grid( row = 0, column = 1 )

sizeY = tkinter.Entry( volPanel, state = 'disabled' )
sizeY.grid( row = 1, column = 1 )

sizeZ = tkinter.Entry( volPanel, state = 'disabled' )
sizeZ.grid( row = 2, column = 1 )

midpointX = tkinter.Entry( volPanel )
midpointX.insert( tkinter.END, '0' )
midpointX.config( state = 'disabled' )
midpointX.grid( row = 3, column = 1 )

midpointY = tkinter.Entry( volPanel )
midpointY.insert( tkinter.END, '0' )
midpointY.config( state = 'disabled' )
midpointY.grid( row = 4, column = 1  )

midpointZ = tkinter.Entry( volPanel )
midpointZ.insert( tkinter.END, '0' )
midpointZ.config( state = 'disabled' )
midpointZ.grid( row = 5, column = 1 )

voxelSizeX = tkinter.Entry( volPanel, state = 'disabled' )
voxelSizeX.grid( row = 6, column = 1 )

voxelSizeY = tkinter.Entry( volPanel, state = 'disabled' )
voxelSizeY.grid( row = 7, column = 1 )

voxelSizeZ = tkinter.Entry( volPanel, state = 'disabled' )
voxelSizeZ.grid( row = 8, column = 1 )

output = tkinter.Entry( volPanel )
output.insert( tkinter.END, 'reconstruction.raw' )
output.config( state = 'disabled' )
output.grid( row = 9, column = 1 )

algoPanel = tkinter.Frame( parameters )
algoPanel.grid( row = 0, column = 3 )

parameters.add( algoPanel, text = 'Algorithms' )

tkinter.Label( algoPanel, text = 'Algorithm' ).grid(
    row = 0, column = 0, sticky = tkinter.W )
tkinter.Label( algoPanel, text = 'Filter' ).grid(
    row = 1, column = 0, sticky = tkinter.W )
tkinter.Label( algoPanel, text = 'Iterations' ).grid(
    row = 2, column = 0, sticky = tkinter.W )

algoList = [
    'FBP',
    'SIRT',
    'CGLS',
    'DIRECTT' ]

algoVar = tkinter.StringVar()
algoVar.set( algoList[ 0 ] )

algorithm = algoList[ 0 ]

algoDict = {
    'FBP': 'BP3D_CUDA',
    'SIRT': 'SIRT3D_CUDA',
    'CGLS': 'CGLS3D_CUDA' }

algoMenu = tkinter.OptionMenu(
    algoPanel, algoVar, *algoList, command = algoF )
algoMenu.config( state = 'disabled' )
algoMenu.grid( row = 0, column = 1, sticky = tkinter.W )

filterList = [
    'Ram-Lak',
    'Shepp-Logan',
    'none' ]

filterVar = tkinter.StringVar()
filterVar.set( filterList[ 0 ] )

window = filterList[ 0 ]

filterMenu = tkinter.OptionMenu(
    algoPanel, filterVar, *filterList, command = windowF )
filterMenu.config( state = 'disabled' )
filterMenu.grid( row = 1, column = 1, sticky = tkinter.W )

iterations = tkinter.Entry( algoPanel )
iterations.insert( tkinter.END, '1' )
iterations.config( state = 'disabled' )
iterations.grid( row = 2, column = 1 )

slices = tkinter.ttk.Notebook( root, height = 655, width = 615 )
slices.grid( row = 0, column = 2, padx = 5 )

projSlices = tkinter.Frame( slices )
projSlices.grid( row = 0, column = 0 )

slices.add( projSlices, text = 'Projections' )

volSlices = tkinter.Frame( slices )
volSlices.grid( row = 0, column = 1 )

slices.add( volSlices, text = 'Volume', state = 'disabled' )

loadButton = tkinter.Button(
    root, text = 'Load dataset',font = ( 'Verdana', 11 ), command = load )
loadButton.grid( row = 1, column = 0, pady = 10 )

recButton = tkinter.Button( root, text = 'Reconstruct', font = (
    'Verdana', 11 ), command = reconstruct )
recButton.config( state = 'disabled' )
recButton.grid( row = 1, column = 1 )

saveButton = tkinter.Button(
    root, text = 'Save volume', font = ( 'Verdana', 11 ), command = save )
saveButton.config( state = 'disabled' )
saveButton.grid( row = 1, column = 2 )

root.mainloop()
