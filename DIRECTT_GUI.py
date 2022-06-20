import tkinter
import tkinter.filedialog
import tkinter.ttk
import PIL.Image
import PIL.ImageTk
import numpy as np
import scipy.signal
import imageio
import astra
import skimage.filters

def VolRead():
    
    global ProjData, ProjCount, DetectorRowCount, DetectorColCount    
    global DetectorOffsetU, DetectorOffsetV
    global GridColCount, GridRowCount, GridSliceCount, ProjHeight, ProjWidth
    global PhotoImageProj, ProjLabel, ProjScale
    
    filename = tkinter.filedialog.askopenfilename(
        initialdir = 'C:/', title = 'Select a file' )
    
    root.title( filename )
    
    ProjData = imageio.volread( filename )
    
    ProjCount.grid_forget()           
    ProjCount = tkinter.Entry( Proj, font = ( 'Verdana', 9 ) )
    ProjCount.insert( tkinter.END, str( ProjData.shape[ 0 ] ) )
    ProjCount.config( state = 'readonly' )
    ProjCount.grid( row = 0, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )
    
    DetectorRowCount.grid_forget()    
    DetectorRowCount = tkinter.Entry( Proj, font = ( 'Verdana', 9 ) )
    DetectorRowCount.insert( tkinter.END, str( ProjData.shape[ 1 ] ) )
    DetectorRowCount.config( state = 'readonly' )
    DetectorRowCount.grid( row = 1, column = 1, pady = (
        0, 3 ), sticky = tkinter.S )
    
    DetectorColCount.grid_forget()
    DetectorColCount = tkinter.Entry( Proj, font = ( 'Verdana', 9 ) )
    DetectorColCount.insert( tkinter.END, str( ProjData.shape[ 2 ] ) )
    DetectorColCount.config( state = 'readonly' )
    DetectorColCount.grid( row = 2, column = 1, pady = (
        0, 3 ), sticky = tkinter.S )
    
    ProjPixelSizeX.config( state = 'normal' )
    ProjPixelSizeY.config( state = 'normal' )
    
    ProjMatrixStartAngle.config( state = 'normal' )
    
    Beam.config( state = 'normal' )
    
    DetectorOffsetU.grid_forget()    
    DetectorOffsetU = tkinter.Entry( Geo, font = ( 'Verdana', 9 ) )
    DetectorOffsetU.insert( tkinter.END, str( ProjData.shape[ 2 ] / 2 - .5 ) )
    DetectorOffsetU.grid( row = 4, column = 1, pady = (
        0, 3 ), sticky = tkinter.SW )
    
    DetectorOffsetV.grid_forget()
    DetectorOffsetV = tkinter.Entry( Geo, font = ( 'Verdana', 9 ) )
    DetectorOffsetV.insert( tkinter.END, str( ProjData.shape[ 1 ] / 2 - .5 ) )
    DetectorOffsetV.grid( row = 5, column = 1, pady = (
        0, 3 ), sticky = tkinter.SW )
    
    ScanAngle.config( state = 'normal' )
    
    AcquisitionDirection.config( state = 'normal' )
    
    a.config( state = 'normal' )
    b.config( state = 'normal' )
    c.config( state = 'normal' )
    
    GridColCount.grid_forget()    
    GridColCount = tkinter.Entry( Vol, font = ( 'Verdana', 9 ) )
    GridColCount.insert( tkinter.END, DetectorColCount.get() )
    GridColCount.grid( row = 0, column = 1, pady = (
        0, 3 ), sticky = tkinter.S )
    
    GridRowCount.grid_forget()    
    GridRowCount = tkinter.Entry( Vol, font = ( 'Verdana', 9 ) )
    GridRowCount.insert( tkinter.END, DetectorColCount.get() )
    GridRowCount.grid( row = 1, column = 1, pady = (
        0, 3 ), sticky = tkinter.S )
    
    GridSliceCount.grid_forget()
    GridSliceCount = tkinter.Entry( Vol, font = ( 'Verdana', 9 ) )
    GridSliceCount.insert( tkinter.END, DetectorRowCount.get() )
    GridSliceCount.grid( row = 2, column = 1, pady = (
        0, 3 ), sticky = tkinter.S )
    
    MidpointX.config( state = 'normal' )
    MidpointY.config( state = 'normal' )
    MidpointZ.config( state = 'normal' )
    
    VoxelSizeX.config( state = 'normal' )
    VoxelSizeY.config( state = 'normal' )
    VoxelSizeZ.config( state = 'normal' )
    
    VolumeOutputFile.config( state = 'normal' )
    
    AlgoMenu.config( state = 'normal' )
    
    FilterMenu.config( state = 'normal' )
    
    if np.any( ProjData <= 0 ):
        
        ProjData[ ProjData <= 0 ] = ProjData[ ProjData > 0 ].min()
    
    ProjData = - np.log( ProjData ).astype( np.float32 )
    ProjData = ProjData.transpose( 1, 0, 2 )
    
    if ProjData.shape[ 0 ] > ProjData.shape[ 2 ]:
        
        ProjHeight = 640
        
        ProjWidth = np.ceil( 640 * ProjData.shape[ 2 ] / ProjData.shape[
            0 ] ).astype( int )   
    
    else:
        
        ProjWidth = 640
        
        ProjHeight = np.ceil( 640 * ProjData.shape[ 0 ] / ProjData.shape[
            2 ] ).astype( int )
    
    PhotoImageProj = PIL.ImageTk.PhotoImage( image = PIL.Image.fromarray(
        np.uint8( ( ProjData[ :, ( int(
        ProjCount.get() ) + 1 ) // 2 - 1 ] - ProjData.min() ) / (
        ProjData.max() - ProjData.min() ) * 255 ) ).resize( (
        ProjWidth, ProjHeight ) ) )
    
    ProjLabel = tkinter.Label( ProjSlices, image = PhotoImageProj )
    ProjLabel.grid( row = 0, pady = 15 )
    
    ProjIntVar = tkinter.IntVar()
    ProjIntVar.set( ( int( ProjCount.get() ) - 1 ) // 2 + 1 )
    
    ProjScale = tkinter.Scale(
        ProjSlices, variable = ProjIntVar, from_ = 1, to = int(
        ProjCount.get() ), orient = tkinter.HORIZONTAL, length = 640,
        command = ProjScaleFunction )
    ProjScale.grid( row = 1 )
    
    Reconstruct.config( state = 'normal' )

def GeoFunction( var ):
    
    global SourceObjectDistance, SourceImageDistance, ScanAngle, GeoKey
    
    GeoKey = var
    
    ScanAngle.grid_forget()
    
    if var == 'Parallel':
        
        SourceObjectDistance.grid_forget()
        SourceObjectDistance = tkinter.Entry( Geo, font = ( 'Verdana', 9 ) )
        SourceObjectDistance.insert( tkinter.END, '1' )
        SourceObjectDistance.config( state = 'disabled' )
        SourceObjectDistance.grid( row = 2, column = 1, pady = (
            0, 3 ), sticky = tkinter.SW )
        
        SourceImageDistance.grid_forget()
        SourceImageDistance = tkinter.Entry( Geo, font = ( 'Verdana', 9 ) )
        SourceImageDistance.insert( tkinter.END, '0' )
        SourceImageDistance.config( state = 'disabled' )
        SourceImageDistance.grid( row = 3, column = 1, pady = (
            0, 3 ), sticky = tkinter.SW )
    
        ScanAngle = tkinter.Entry( Geo, font = ( 'Verdana', 9 ) )
        ScanAngle.insert( tkinter.END, '360' )
        ScanAngle.grid( row = 6, column = 1, pady = (
            0, 3 ), sticky = tkinter.SW )
    
    else:
        
        SourceObjectDistance.config( state = 'normal' )
        
        SourceImageDistance.config( state = 'normal' )
        
        ScanAngle = tkinter.Entry( Geo, font = ( 'Verdana', 9 ) )
        ScanAngle.insert( tkinter.END, '360' )
        ScanAngle.grid( row = 6, column = 1, pady = (
            0, 3 ), sticky = tkinter.SW )

def AcquisitionDirectionFunction( var ):
    
    global clockwise
    
    clockwise = var == 'Clockwise'

def AlgoFunction( var ):
    
    global AlgoKey, FilterMenu, Filter
    global IterationsCountMenu, IterationsCountKey, IterationsCount
    
    AlgoKey = var
    
    if var == AlgoList[ 0 ]:
        
        FilterVar.set( FilterList[ 0 ] )
        
        FilterMenu.config( state = 'normal' )
        
        Filter = 'Ram-Lak'
        
        IterationsCountMenu.config( state = 'disabled' )
        
        IterationsCount.config( state = 'disabled' )
    
    elif var in AlgoList[ 1:3 ]:
        
        FilterVar.set( FilterList[ -1 ] )
        
        FilterMenu.config( state = 'disabled' )
        
        Filter = 'none'
        
        IterationsCountMenu.config( state = 'disabled' )
        
        IterationsCount.config( state = 'normal' )
    
    else:
        
        FilterVar.set( FilterList[ -1 ] )
        
        FilterMenu.config( state = 'disabled' )
        
        Filter = 'none'
        
        IterationsCountVar.set( IterationsCountList[ 0 ] )
        
        IterationsCountMenu.config( state = 'normal' )
        
        IterationsCountKey = 'Set manually'
        
        IterationsCount.config( state = 'normal' )

def FilterFunction( var ):
    
    global Filter
    
    Filter = var

def IterationsCountFunction( var ):
    
    global IterationsCountKey
    
    IterationsCountKey = var
    
    if var == 'Set manually':
        
        IterationsCount.config( state = 'normal' )
    
    else:
        
        IterationsCount.config( state = 'disabled' )

def ProjScaleFunction( var ):
    
    global ProjLabel, PhotoImageProj
    
    ProjLabel.grid_forget()
    
    PhotoImageProj = PIL.ImageTk.PhotoImage( image = PIL.Image.fromarray(
        np.uint8( ( ProjData[ :, int( var ) - 1 ] - ProjData.min() ) / (
        ProjData.max() - ProjData.min() ) * 255 ) ).resize( (
        ProjWidth, ProjHeight ) ) )
    
    ProjLabel = tkinter.Label( ProjSlices, image = PhotoImageProj )
    ProjLabel.grid( row = 0, pady = 15 )

def VolScaleFunction( var ):
    
    global VolLabel, PhotoImageVol
    
    VolLabel.grid_forget()
    
    PhotoImageVol = PIL.ImageTk.PhotoImage( image = PIL.Image.fromarray(
        ReconstructionData[ int( var ) - 1 ] ).resize( (
        VolWidth, VolHeight ) ) )
    
    VolLabel = tkinter.Label( VolSlices, image = PhotoImageVol )
    VolLabel.grid( row = 0, pady = 15 )

def Run():
    
    global ThresholdTriangle, ReconstructionData
    global Radon, RadonColLeft, RadonColRight, RadonRowTop, RadonRowBottom
    global VolWidth, VolHeight, VolLabel, VolScale, PhotoImageVol
    
    ExtraGridSlices = int( int( GridSliceCount.get() ) * np.amax( [ int(
        GridColCount.get() ) * float( VoxelSizeX.get() ), int(
        GridRowCount.get() ) * float( VoxelSizeY.get() ) ] ) / ( 2 * float(
        SourceObjectDistance.get() ) ) ) * ( GeoKey == 'Cone' )
    ExtraGridSlices -= ExtraGridSlices % 2
    
    theta = np.linspace( np.radians( float(
        ProjMatrixStartAngle.get() ) ), np.radians( float(
        ProjMatrixStartAngle.get() ) + float( ScanAngle.get() ) ) * (
        1 - 2 * clockwise ), int( ProjCount.get() ), False )
    
    Vectors = np.zeros( [ int( ProjCount.get() ), 12 ] )
    Vectors[ :, 0 ] = np.sin( theta ) * ( ( GeoKey == 'Parallel' ) + float(
        SourceImageDistance.get() ) / float( ProjPixelSizeX.get() ) )
    Vectors[ :, 1 ] = - np.cos( theta ) * ( ( GeoKey == 'Parallel' ) + float(
        SourceImageDistance.get() ) / float( ProjPixelSizeX.get() ) )
    Vectors[ :, 3 ] = np.cos( theta ) * ( int(
        DetectorColCount.get() ) / 2 - .5 - float(  DetectorOffsetU.get() ) )
    Vectors[ :, 4 ] = np.sin( theta ) * ( int(
        DetectorColCount.get() ) / 2 - .5 - float( DetectorOffsetU.get() ) )
    Vectors[ :, 5 ] = float( DetectorOffsetV.get() ) - int(
        DetectorRowCount.get() ) / 2 + .5 
    Vectors[ :, 6 ] = np.cos( theta ) * np.cos( np.radians( float(
        b.get() ) ) ) * np.cos( np.radians( float( c.get() ) ) )
    Vectors[ :, 7 ] = np.sin( theta ) * np.cos( np.radians( float(
        b.get() ) ) ) * np.cos( np.radians( float( c.get() ) ) )
    Vectors[ :, 8 ] = np.sin( np.radians( float( b.get() ) ) )
    Vectors[ :, 9 ] = np.cos( theta ) * np.sin( np.radians( float(
        b.get() ) ) ) * float( ProjPixelSizeY.get() ) / float(
        ProjPixelSizeX.get() )
    Vectors[ :, 10 ] = np.sin( theta ) * np.sin( np.radians( float(
        b.get() ) ) ) * float( ProjPixelSizeY.get() ) / float(
        ProjPixelSizeX.get() )
    Vectors[ :, 11 ] = np.cos( np.radians( float( a.get() ) ) ) * np.cos(
        np.radians( float( b.get() ) ) ) * float(
        ProjPixelSizeY.get() ) / float( ProjPixelSizeX.get() )
    
    ProjGeom = astra.create_proj_geom( GeoDict[ GeoKey ][ 1 ], int(
        DetectorRowCount.get() ), int( DetectorColCount.get() ), Vectors )
    
    VolGeom = astra.create_vol_geom(
        int( GridRowCount.get() ), int( GridColCount.get() ),
        int( GridSliceCount.get() ) + ExtraGridSlices,
        float( MidpointX.get() ) / float( VoxelSizeX.get() ) - int(
            GridColCount.get() ) / 2,
        float( MidpointX.get() ) / float( VoxelSizeX.get() ) + int(
            GridColCount.get() ) / 2,
        float( MidpointY.get() ) / float( VoxelSizeY.get() ) - int(
            GridRowCount.get() ) / 2,
        float( MidpointY.get() ) / float( VoxelSizeY.get() ) + int(
            GridRowCount.get() ) / 2,
        float( MidpointZ.get() ) / float( VoxelSizeZ.get() ) - ( int(
            GridSliceCount.get() ) + ExtraGridSlices ) / 2,
        float( MidpointZ.get() ) / float( VoxelSizeZ.get() ) + ( int(
            GridSliceCount.get() ) + ExtraGridSlices ) / 2 )
    
    ReconstructionDataId = astra.data3d.create( '-vol', VolGeom )
    
    AstraDict = astra.astra_dict( AlgoDict[ AlgoKey ] )
    AstraDict[ 'ReconstructionDataId' ] = ReconstructionDataId
    
    if AlgoKey == 'DIRECTT':
        
        ThresholdTriangle = skimage.filters.threshold_triangle( ProjData )
        
        SpacedX = np.linspace( - int( GridColCount.get() ) / 2 + .5, int(
            GridColCount.get() ) / 2 - .5, int( GridColCount.get() ) )
        
        SpacedY = np.linspace( - int( GridRowCount.get() ) / 2 + .5, int(
            GridRowCount.get() ) / 2 - .5, int( GridRowCount.get() ) )
        
        X, Y = np.meshgrid( SpacedX, SpacedY )
        
        Ellipse = np.array( ( X / ( int(
            GridColCount.get() ) / 2 - .5 ) ) ** 2 + ( Y / ( int(
            GridRowCount.get() ) / 2 - .5 ) ) ** 2 < 1 )
        
        Phantom = Ellipse.reshape( 1, int( GridRowCount.get() ), int(
            GridColCount.get() ) ) * np.ones( [ int(
            GridSliceCount.get() ) + ExtraGridSlices, int(
            GridRowCount.get() ), int(
            GridColCount.get() ) ], dtype = np.int8 )
        
        PhantomProjDataId, PhantomProjData = astra.create_sino3d_gpu(
            Phantom, ProjGeom, VolGeom )
        
        astra.data3d.delete( [ PhantomProjDataId ] )
        
        del Phantom
        
        PhantomProjDataAxis0 = PhantomProjData.reshape( int(
            DetectorRowCount.get() ) * int( ProjCount.get() ), int(
            DetectorColCount.get() ) )
        
        DetectorColLeft = np.argmax( np.any(
            PhantomProjDataAxis0 > 0, axis = 0 ) )
        
        DetectorColRight = np.argmax( np.any( PhantomProjDataAxis0[
            :, -1::-1 ] > 0, axis = 0 ) )
        
        ProjGeom[ 'Vectors' ][ :, 3 ] += np.cos( theta ) * (
            DetectorColLeft - DetectorColRight ) / 2
        ProjGeom[ 'Vectors' ][ :, 4 ] += np.sin( theta ) * (
            DetectorColLeft - DetectorColRight ) / 2
        
        DetectorColRight = int( DetectorColCount.get() ) - DetectorColRight
        
        ProjGeom[ 'DetectorColCount' ] = ProjData[
            ..., DetectorColLeft:DetectorColRight ].shape[ 2 ]
        
        RadonColLeft = np.argmax( np.all(
            PhantomProjDataAxis0 > 0, axis = 0 ) ) - DetectorColLeft
        
        RadonColRight = int( DetectorColCount.get() ) - np.argmax( np.all(
            PhantomProjDataAxis0[
            :, -1::-1 ] > 0, axis = 0 ) ) - DetectorColLeft
        
        del PhantomProjDataAxis0
        
        PhantomProjDataAxis1 = PhantomProjData.reshape( int(
            DetectorRowCount.get() ), int( ProjCount.get() ) * int(
            DetectorColCount.get() ) )
        
        DetectorRowTop = np.argmax( np.any(
            PhantomProjDataAxis1 > 0, axis = 1 ) )
        
        DetectorRowBottom = np.argmax( np.any( PhantomProjDataAxis1[
            -1::-1 ] > 0, axis = 1 ) )
        
        ProjGeom[ 'Vectors' ][ :, 5 ] += (
            DetectorRowTop - DetectorRowBottom ) / 2
        
        DetectorRowBottom = int( DetectorRowCount.get() ) - DetectorRowBottom
        
        ProjGeom[ 'DetectorRowCount' ] = ProjData[
            DetectorRowTop:DetectorRowBottom ].shape[ 0 ]
        
        RadonRowTop = np.argmax( np.all(
            PhantomProjDataAxis1 > 0, axis = 1 ) ) - DetectorRowTop
        
        RadonRowBottom = int( DetectorRowCount.get() ) - np.argmax( np.all(
            PhantomProjDataAxis1[ -1::-1 ] > 0, axis = 1 ) ) - DetectorRowTop
        
        del PhantomProjDataAxis1
        
        PhantomProjData *= ProjData > ThresholdTriangle
        
        Norm = np.linalg.norm( ProjData[ DetectorRowTop:DetectorRowBottom, :,
            DetectorColLeft:DetectorColRight ] )
        
        beta1 = Norm / np.linalg.norm( PhantomProjData )
        
        ProjGeom[ 'DetectorRowCount' ] = 1
        ProjGeom[ 'Vectors' ][ :, 5 ] = 0
        
        VolGeom[ 'option' ][ 'WindowMinZ' ] = -.5
        VolGeom[ 'option' ][ 'WindowMaxZ' ] = .5
        VolGeom[ 'GridSliceCount' ] = 1
        
        EllipsisProjDataId, EllipsisProjData = astra.create_sino3d_gpu(
            Ellipse.reshape( 1, int( GridRowCount.get() ), int(
            GridColCount.get() ) ), ProjGeom, VolGeom )
        
        SourcePlaneId = astra.data3d.create( '-vol', VolGeom )
        
        SourcePlaneDict = astra.astra_dict( 'BP3D_CUDA' )
        SourcePlaneDict[ 'ReconstructionDataId' ] = SourcePlaneId
        SourcePlaneDict[ 'ProjectionDataId' ] = EllipsisProjDataId
        
        PhantomProjData[
            ..., DetectorColLeft:DetectorColRight ] /= EllipsisProjData
        
        AlgorithmId = astra.algorithm.create( SourcePlaneDict )
        
        astra.algorithm.run( AlgorithmId )
        
        SourcePlane = astra.data3d.get( SourcePlaneId )
        
        SourcePlaneMax = SourcePlane.max()
        
        astra.data3d.delete( [ EllipsisProjDataId ] )
        
        astra.algorithm.delete( AlgorithmId )
        
        EllipsisProjDataId = astra.data3d.create(
            '-sino', ProjGeom, np.ones(
            EllipsisProjData.shape, dtype = np.int8 ) )
        
        SourcePlaneDict[ 'ProjectionDataId' ] = EllipsisProjDataId
        
        AlgorithmId = astra.algorithm.create( SourcePlaneDict )
        
        astra.algorithm.run( AlgorithmId )
        
        SourcePlane = astra.data3d.get( SourcePlaneId )
        
        astra.data3d.delete( [ EllipsisProjDataId, SourcePlaneId ] )
        
        astra.algorithm.delete( AlgorithmId )
        
        ProjGeom[ 'DetectorRowCount' ] = ProjData[
            DetectorRowTop:DetectorRowBottom ].shape[ 0 ]
        ProjGeom[ 'Vectors' ][ :, 5 ] = Vectors[ :, 5 ]
        
        VolGeom[ 'option' ][ 'WindowMinZ' ] = float(
            MidpointZ.get() ) / float( VoxelSizeZ.get() ) - ( int(
            GridSliceCount.get() ) + ExtraGridSlices ) / 2
        VolGeom[ 'option' ][ 'WindowMaxZ' ] = float(
            MidpointZ.get() ) / float( VoxelSizeZ.get() ) + ( int(
            GridSliceCount.get() ) + ExtraGridSlices ) / 2
        VolGeom[ 'GridSliceCount' ] = int(
            GridSliceCount.get() ) + ExtraGridSlices
        
        PhantomProjDataId = astra.data3d.create(
            '-sino', ProjGeom, PhantomProjData[
            DetectorRowTop:DetectorRowBottom, :,
            DetectorColLeft:DetectorColRight ] )
        
        AstraDict[ 'ProjectionDataId' ] = PhantomProjDataId
        
        AlgorithmId = astra.algorithm.create( AstraDict )
        
        astra.algorithm.run( AlgorithmId )
        
        del PhantomProjData
        
        M = astra.data3d.get( ReconstructionDataId )
        M /= SourcePlane
        
        astra.data3d.delete( [ PhantomProjDataId ] )
        
        astra.algorithm.delete( AlgorithmId )
        
        ProjDataId = astra.data3d.create( '-sino', ProjGeom, ProjData[
            DetectorRowTop:DetectorRowBottom, :,
            DetectorColLeft:DetectorColRight ] )
        
        AstraDict[ 'ProjectionDataId' ] = ProjDataId
        
        AlgorithmId = astra.algorithm.create( AstraDict )
        
        astra.algorithm.run( AlgorithmId )
        
        ReconstructionData = astra.data3d.get( ReconstructionDataId )
        ReconstructionData /= SourcePlaneMax
        ReconstructionData[ M > 0 ] /= M[ M > 0 ]
        
        Selection = ReconstructionData[ M >= 1 ].max() - beta1
        
        ReconstructionData -= ReconstructionData[ M >= 1 ].max() - Selection
        ReconstructionData *= ReconstructionData > 0
        
        alpha = int( beta1 / ReconstructionData[ M >= 1 ].max() )
        
        ReconstructionData *= alpha
        
        RadonDataId, RadonData = astra.create_sino3d_gpu(
            ReconstructionData, ProjGeom, VolGeom )
        
        astra.data3d.delete( [ ProjDataId, RadonDataId ] )
        
        astra.algorithm.delete( AlgorithmId )
        
        k = 1
        
        while Repeat( k, RadonData[
            RadonRowTop:RadonRowBottom, :, RadonColLeft:RadonColRight ] ):
            
            ResData = ProjData[ DetectorRowTop:DetectorRowBottom, :,
                DetectorColLeft:DetectorColRight ] - RadonData
            
            ResDataId = astra.data3d.create( '-sino', ProjGeom, ResData )
            
            AstraDict[ 'ProjectionDataId' ] = ResDataId
            
            AlgorithmId = astra.algorithm.create( AstraDict )
            
            astra.algorithm.run( AlgorithmId )
            
            BackprojData = astra.data3d.get( ReconstructionDataId )
            BackprojData /= SourcePlaneMax
            BackprojData[ M > 0 ] /= M[ M > 0 ]
            BackprojData -= BackprojData[ M >= 1 ].max() - np.linalg.norm(
                ResData ) / Norm * Selection
            BackprojData *= BackprojData > 0
            
            ReconstructionData += alpha * BackprojData
            
            RadonDataId, RadonData = astra.create_sino3d_gpu(
                ReconstructionData, ProjGeom, VolGeom )
            
            astra.data3d.delete( [ ResDataId, RadonDataId ] )
            
            astra.algorithm.delete( AlgorithmId )
            
            k += 1
    
    else:
        
        ProjDataId = astra.data3d.create( '-sino', ProjGeom, Convolve(
            ProjData, int( DetectorColCount.get() ) ) )
        
        AstraDict[ 'ProjectionDataId' ] = ProjDataId
        
        AlgorithmId = astra.algorithm.create( AstraDict )
        
        astra.algorithm.run( AlgorithmId, int( IterationsCount.get() ) )
        
        ReconstructionData = astra.data3d.get( ReconstructionDataId )
        
        if AlgoKey == 'FBP':
            
            ReconstructionData *= abs( theta[ 1 ] - theta[ 0 ] )
        
        astra.data3d.delete( ProjDataId )
        
        astra.algorithm.delete( AlgorithmId )
    
    ReconstructionData = ReconstructionData[ ( ExtraGridSlices // 2 ):(
        ExtraGridSlices // 2 + int( GridSliceCount.get() ) ) ]
    ReconstructionData = skimage.img_as_uint( ReconstructionData )
    
    if AlgoKey in AlgoList[ 1:3 ]:
        
        print( '\n', AlgoKey, 'was terminated after ' +
              IterationsCount.get() + ' iterations.' )
    
    elif AlgoKey == 'DIRECTT':
        
        print( '\nDIRECTT was terminated after', k, 'iterations.' )
    
    if int( GridColCount.get() ) > int( GridRowCount.get() ):
        
        VolWidth = 800
        
        VolHeight = np.ceil( 800 * int( GridRowCount.get() ) / int(
            GridColCount.get() ) ).astype( int )
    
    else:
        
        VolHeight = 800
        
        VolWidth = np.ceil( 800 * int( GridColCount.get() ) / int(
            GridRowCount.get() ) ).astype( int )
    
    Slices.add( VolSlices, text = 'Volume', state = 'normal' )
    
    PhotoImageVol = PIL.ImageTk.PhotoImage( image = PIL.Image.fromarray(
        ReconstructionData[ ( int(
        GridSliceCount.get() ) + 1 ) // 2 - 1 ] ).resize( (
        VolWidth, VolHeight ) ) )
    
    VolLabel = tkinter.Label( VolSlices, image = PhotoImageVol )
    VolLabel.grid( row = 0, pady = 15 )
    
    VolIntVar = tkinter.IntVar()
    VolIntVar.set( ( int( GridSliceCount.get() ) - 1 ) // 2 + 1 )
    
    VolScale = tkinter.Scale(
        VolSlices, variable = VolIntVar, from_ = 1, to = int(
        GridSliceCount.get() ), orient = tkinter.HORIZONTAL, length = 800,
        command = VolScaleFunction )
    VolScale.grid( row = 1 )
    
    SaveVolume.config( state = 'normal' )

def Repeat( Iterations, ForwardProj ):
    
    if IterationsCountKey == 'Set manually':
        
        return Iterations < int( IterationsCount.get() )
    
    else:
        
        return np.any( ForwardProj[ ProjData[
            RadonRowTop:RadonRowBottom, :, RadonColLeft:RadonColRight
            ] > ThresholdTriangle ] == 0 )

def Convolve( sino, s ):
    
    if Filter == 'Ram-Lak':
        
        rampbl = np.zeros( s * 2 - s % 2, dtype = np.float32 )    
        rampbl[ s - 1 ] = .25
        
        idxodd = np.concatenate( ( np.flip( -1 * np.arange(
            1, rampbl.size // 2, 2 ), axis = 0 ), np.arange(
            1, rampbl.size // 2, 2 ) ) )
        
        rampbl[ ( s % 2 )::2 ] = -1 / ( idxodd * np.pi ) ** 2
        
        return scipy.signal.convolve( sino, rampbl.reshape( [
            1, 1, rampbl.size ] ), mode = 'same' )
    
    elif Filter == 'Shepp-Logan':
        
        rampbl = -2 / np.pi ** 2 / ( 4 * np.arange( - s + 1, s + (
            s + 1 ) % 2 ) ** 2 - 1 )
        
        return scipy.signal.convolve( sino, rampbl.reshape( [
            1, 1, rampbl.size ] ), mode = 'same' )
    
    else:
        
        return sino

def ToFile():
    
    Directory = tkinter.filedialog.askdirectory(
        initialdir = 'C:/', title = 'Select a folder' )
    
    ReconstructionData.tofile( Directory + '/' + VolumeOutputFile.get() )

root = tkinter.Tk()

Parameters = tkinter.ttk.Notebook( root, height = 720, width = 640 )
Parameters.grid( row = 0, column = 0, columnspan = 2 )

Proj = tkinter.Frame( Parameters, pady = 9 )
Proj.grid( row = 0, column = 0 )

Parameters.add( Proj, text = ' Projections ' )

tkinter.Label( Proj, text = 'Projection count (circular scan)', font = (
    'Verdana',  9 ) ).grid(
    row = 0, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Proj, text = 'Size X', font = ( 'Verdana', 9 ) ).grid(
    row = 1, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Proj, text = 'Size Y', font = ( 'Verdana', 9 ) ).grid(
    row = 2, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Proj, text = 'Projection pixel size X', font = (
    'Verdana',  9 ) ).grid(
    row = 3, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Proj, text = 'Projection pixel size Y', font = (
    'Verdana',  9 ) ).grid(
    row = 4, column = 0, padx = 9, sticky = tkinter.W )

ProjCount = tkinter.Entry( Proj, state = 'disabled', font = ( 'Verdana', 9 ) )
ProjCount.grid( row = 0, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )

DetectorColCount = tkinter.Entry( Proj, state = 'disabled', font = (
    'Verdana', 9 ) )
DetectorColCount.grid( row = 1, column = 1, pady = (
    0, 3 ), sticky = tkinter.S )

DetectorRowCount = tkinter.Entry( Proj, state = 'disabled', font = (
    'Verdana', 9 ) )
DetectorRowCount.grid( row = 2, column = 1, pady = (
    0, 3 ), sticky = tkinter.S )

ProjPixelSizeX = tkinter.Entry( Proj, state = 'disabled', font = (
    'Verdana', 9 ) )
ProjPixelSizeX.grid( row = 3, column = 1, pady = (
    0, 3 ), sticky = tkinter.S )

ProjPixelSizeY = tkinter.Entry( Proj, state = 'disabled', font = (
    'Verdana', 9 ) )
ProjPixelSizeY.grid( row = 4, column = 1, pady = (
    0, 3 ), sticky = tkinter.S )

Geo = tkinter.Frame( Parameters, pady = 9 )
Geo.grid( row = 0, column = 1 )

Parameters.add( Geo, text = ' Geometry ' )

tkinter.Label( Geo, text = 'Projection matrix start angle', font = (
    'Verdana', 9 ) ).grid(
    row = 0, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'Beam', font = ( 'Verdana', 9 ) ).grid(
    row = 1, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'Source object distance', font = (
    'Verdana', 9 ) ).grid(
    row = 2, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'Source image distance', font = (
    'Verdana', 9 ) ).grid(
    row = 3, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'Detector offset u', font = (
    'Verdana', 9 ) ).grid(
    row = 4, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'Detector offset v', font = (
    'Verdana', 9 ) ).grid(
    row = 5, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'Scan angle', font = ( 'Verdana', 9 ) ).grid(
    row = 6, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'Acquisition direction', font = (
    'Verdana', 9 ) ).grid(
    row = 7, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'a', font = ( 'Verdana', 9 ) ).grid(
    row = 8, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Geo, text = 'b', font = ( 'Verdana', 9 ) ).grid(
    row = 9, column = 0, padx = 9, sticky = tkinter.W  )
tkinter.Label( Geo, text = 'c', font = ( 'Verdana', 9 ) ).grid(
    row = 10, column = 0, padx = 9, sticky = tkinter.W )

ProjMatrixStartAngle = tkinter.Entry( Geo )
ProjMatrixStartAngle.insert( tkinter.END, '0' )
ProjMatrixStartAngle.config( state = 'disabled', font = ( 'Verdana', 9 ) )
ProjMatrixStartAngle.grid( row = 0, column = 1, pady = (
    0, 3 ), sticky = tkinter.SW )

ΒeamList = [
    'Parallel',
    'Cone' ]

ΒeamVar = tkinter.StringVar( Geo )
ΒeamVar.set( ΒeamList[ 0 ] )

GeoKey = 'Parallel'

GeoDict = {
    'Parallel': [ 0, 'parallel3d_vec' ],
    'Cone': [ 1, 'cone_vec' ] }

Beam = tkinter.OptionMenu( Geo, ΒeamVar, *ΒeamList, command = GeoFunction )
Beam.config( state = 'disabled', font = ( 'Verdana',  9 ) )
Beam.grid( row = 1, column = 1, pady = ( 0, 3 ), sticky = tkinter.SW )

SourceObjectDistance = tkinter.Entry( Geo )
SourceObjectDistance.insert( tkinter.END, '1' )
SourceObjectDistance.config( state = 'disabled', font = ( 'Verdana', 9 ) )
SourceObjectDistance.grid( row = 2, column = 1, pady = (
    0, 3 ), sticky = tkinter.SW)

SourceImageDistance = tkinter.Entry( Geo )
SourceImageDistance.insert( tkinter.END, '0' )
SourceImageDistance.config( state = 'disabled', font = ( 'Verdana', 9 ) )
SourceImageDistance.grid( row = 3, column = 1, pady = (
    0, 3 ), sticky = tkinter.SW )

DetectorOffsetU = tkinter.Entry( Geo, state = 'disabled', font = (
    'Verdana', 9 ) )
DetectorOffsetU.grid( row = 4, column = 1, pady = (
    0, 3 ), sticky = tkinter.SW )

DetectorOffsetV = tkinter.Entry( Geo, state = 'disabled', font = ( 
    'Verdana', 9 ) )
DetectorOffsetV.grid( row = 5, column = 1, pady = (
    0, 3 ), sticky = tkinter.SW )

ScanAngle = tkinter.Entry( Geo )
ScanAngle.insert( tkinter.END, '180' )
ScanAngle.config( state = 'disabled', font = ( 'Verdana', 9 ) )
ScanAngle.grid( row = 6, column = 1, pady = ( 0, 3 ), sticky = tkinter.SW )

AcquisitionDirectionList = [
    'Clockwise',
    'Counter-clockwise' ]

AcquisitionDirectionVar = tkinter.StringVar()
AcquisitionDirectionVar.set( AcquisitionDirectionList[ 1 ] )

clockwise = False

AcquisitionDirection = tkinter.OptionMenu(
    Geo, AcquisitionDirectionVar, *AcquisitionDirectionList,
    command = AcquisitionDirectionFunction )
AcquisitionDirection.config( state = 'disabled', font = (
    'Verdana',  9 ) )
AcquisitionDirection.grid( row = 7, column = 1, pady = (
    0, 3 ), sticky = tkinter.SW )

a = tkinter.Entry( Geo )
a.insert( tkinter.END, '0' )
a.config( state = 'disabled', font = ( 'Verdana', 9 ) )
a.grid( row = 8, column = 1, pady = ( 0, 3 ), sticky = tkinter.SW )

b = tkinter.Entry( Geo )
b.insert( tkinter.END, '0' )
b.config( state = 'disabled', font = ( 'Verdana', 9 ) )
b.grid( row = 9, column = 1, pady = ( 0, 3 ), sticky = tkinter.SW )

c = tkinter.Entry( Geo )
c.insert( tkinter.END, '0' )
c.config( state = 'disabled', font = ( 'Verdana',  9 ) )
c.grid( row = 10, column = 1, pady = ( 0, 3 ), sticky = tkinter.SW )

Vol = tkinter.Frame( Parameters, pady = 9 )
Vol.grid( row = 0, column = 2 )

Parameters.add( Vol, text = ' Volume ' )

tkinter.Label( Vol, text = 'Size X', font = ( 'Verdana', 9 ) ).grid(
    row = 0, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Size Y', font = ( 'Verdana', 9 ) ).grid(
    row = 1, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Size Z', font = ( 'Verdana', 9 ) ).grid(
    row = 2, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Midpoint X', font = ( 'Verdana', 9 ) ).grid(
    row = 3, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Midpoint Y', font = ( 'Verdana', 9 ) ).grid(
    row = 4, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Midpoint Z', font = ( 'Verdana', 9 ) ).grid(
    row = 5, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Voxel size X', font = ( 'Verdana', 9 ) ).grid(
    row = 6, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Voxel size Y', font = ( 'Verdana', 9 ) ).grid(
    row = 7, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Voxel size Z', font = ( 'Verdana', 9 ) ).grid(
    row = 8, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Vol, text = 'Volume output file', font = (
    'Verdana', 9 ) ).grid( row = 9, column = 0, padx = 9, sticky = tkinter.W )

GridColCount = tkinter.Entry( Vol, state = 'disabled', font = (
    'Verdana', 9 ) )
GridColCount.grid( row = 0, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )

GridRowCount = tkinter.Entry( Vol, state = 'disabled', font = (
    'Verdana', 9 ) )
GridRowCount.grid( row = 1, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )

GridSliceCount = tkinter.Entry( Vol, state = 'disabled', font = (
    'Verdana',  9 ) )
GridSliceCount.grid( row = 2, column = 1, pady = (
    0, 3 ), sticky = tkinter.S )

MidpointX = tkinter.Entry( Vol )
MidpointX.insert( tkinter.END, '0' )
MidpointX.config( state = 'disabled', font = ( 'Verdana', 9 ) )
MidpointX.grid( row = 3, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )

MidpointY = tkinter.Entry( Vol )
MidpointY.insert( tkinter.END, '0' )
MidpointY.config( state = 'disabled', font = ( 'Verdana', 9 ) )
MidpointY.grid( row = 4, column = 1, pady = ( 0, 3 ), sticky = tkinter.S  )

MidpointZ = tkinter.Entry( Vol )
MidpointZ.insert( tkinter.END, '0' )
MidpointZ.config( state = 'disabled', font = ( 'Verdana', 9 ) )
MidpointZ.grid( row = 5, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )

VoxelSizeX = tkinter.Entry( Vol, state = 'disabled', font = (
    'Verdana',  9 ) )
VoxelSizeX.grid( row = 6, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )

VoxelSizeY = tkinter.Entry( Vol, state = 'disabled', font = (
    'Verdana',  9 ) )
VoxelSizeY.grid( row = 7, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )

VoxelSizeZ = tkinter.Entry( Vol, state = 'disabled', font = (
    'Verdana',  9 ) )
VoxelSizeZ.grid( row = 8, column = 1, pady = ( 0, 3 ), sticky = tkinter.S )

VolumeOutputFile = tkinter.Entry( Vol )
VolumeOutputFile.insert( tkinter.END, 'volume.raw' )
VolumeOutputFile.config( state = 'disabled', font = ( 'Verdana', 9 ) )
VolumeOutputFile.grid( row = 9, column = 1, pady = (
    0, 3 ), sticky = tkinter.S )

Algo = tkinter.Frame( Parameters, pady = 9 )
Algo.grid( row = 0, column = 3 )

Parameters.add( Algo, text = ' Algorithms ' )

tkinter.Label( Algo, text = 'Algorithm', font = ( 'Verdana', 9 ) ).grid(
    row = 0, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Algo, text = 'Filter', font = ( 'Verdana', 9 ) ).grid(
    row = 1, column = 0, padx = 9, sticky = tkinter.W )
tkinter.Label( Algo, text = 'Iterations count', font = (
    'Verdana', 9 ) ).grid( row = 2, column = 0, padx = 9, sticky = tkinter.W )

AlgoList = [
    'FBP',
    'SIRT',
    'CGLS',
    'DIRECTT' ]

AlgoVar = tkinter.StringVar()
AlgoVar.set( AlgoList[ 0 ] )

AlgoKey = 'FBP'

AlgoDict = {
    'FBP': 'BP3D_CUDA',
    'SIRT': 'SIRT3D_CUDA',
    'CGLS': 'CGLS3D_CUDA',
    'DIRECTT': 'BP3D_CUDA' }

AlgoMenu = tkinter.OptionMenu(
    Algo, AlgoVar, *AlgoList, command = AlgoFunction )
AlgoMenu.config( state = 'disabled', font = ( 'Verdana', 9 ) )
AlgoMenu.grid( row = 0, column = 1, pady = ( 0, 3 ), sticky = tkinter.W )

FilterList = [
    'Ram-Lak',
    'Shepp-Logan',
    'None' ]

FilterVar = tkinter.StringVar()
FilterVar.set( FilterList[ 0 ] )

Filter = 'Ram-Lak'

FilterMenu = tkinter.OptionMenu(
    Algo, FilterVar, *FilterList, command = FilterFunction )
FilterMenu.config( state = 'disabled', font = ( 'Verdana', 9 ) )
FilterMenu.grid( row = 1, column = 1, pady = ( 0, 3 ), sticky = tkinter.W )

IterationsCountList = [
    'Set manually',
    'Set automatically' ]

IterationsCountVar = tkinter.StringVar()
IterationsCountVar.set( IterationsCountList[ 0 ] )

IterationsCountKey = 'Set manually'

IterationsCountMenu = tkinter.OptionMenu(
    Algo, IterationsCountVar, *IterationsCountList,
    command = IterationsCountFunction )
IterationsCountMenu.config( state = 'disabled', font = ( 'Verdana', 9 ) )
IterationsCountMenu.grid( row = 2, column = 1, pady = (
    0, 3 ), sticky = tkinter.W )

IterationsCount = tkinter.Entry( Algo )
IterationsCount.insert( tkinter.END, '1' )
IterationsCount.config( state = 'disabled', font = ( 'Verdana', 9 ) )
IterationsCount.grid( row = 3, column = 1, pady = (
    0, 3 ), sticky = tkinter.W )

Slices = tkinter.ttk.Notebook( root, height = 720, width = 655 )
Slices.grid( row = 0, column = 2, padx = 12.5 )

ProjSlices = tkinter.Frame( Slices )
ProjSlices.grid( row = 0, column = 0 )

Slices.add( ProjSlices, text = ' Projections ' )

VolSlices = tkinter.Frame( Slices )
VolSlices.grid( row = 0, column = 1 )

Slices.add( VolSlices, text = ' Volume ', state = 'disabled' )

LoadData = tkinter.Button( root, text = 'Load data', font = (
    'Verdana', 10 ), command = VolRead )
LoadData.grid( row = 1, column = 0, pady = 15 )

Reconstruct = tkinter.Button( root, text = 'Reconstruct', font = (
    'Verdana', 10 ), command = Run )
Reconstruct.config( state = 'disabled' )
Reconstruct.grid( row = 1, column = 1 )

SaveVolume = tkinter.Button( root, text = 'Save volume', font = (
    'Verdana', 10 ), command = ToFile )
SaveVolume.config( state = 'disabled' )
SaveVolume.grid( row = 1, column = 2 )

root.mainloop()
