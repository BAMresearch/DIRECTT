import numpy as np
import scipy.ndimage
import scipy.optimize
import tkinter
import tkinter.filedialog
import imageio
import PIL.Image
import PIL.ImageTk
import astra

root = tkinter.Tk()
root.title( 'DIRECTT' )
root.filename = tkinter.filedialog.askopenfilename( initialdir = 'C:/', title =
                                                   'Select a file' )
root.title( 'DIRECTT - ' + root.filename )

print( 'Loading', root.filename, '...' )

proj = imageio.volread( root.filename ).transpose( 1, 0, 2 )

detRows, num_angles, detCols = proj.shape

proj_img = PIL.ImageTk.PhotoImage( image = PIL.Image.fromarray( np.uint8( (
        proj[ :, 0, : ] - np.amin( proj ) ) / ( np.amax( proj ) - np.amin( proj
            ) ) * 255 ) ).resize( ( 640, np.ceil( detRows / detCols * 640
            ).astype( int ) ) ) )

root.geometry( '645x' + str( int( detRows / detCols * 640 + 90 ) ) )

projection = tkinter.Label( root, image = proj_img )
projection.grid( row = 0 )

def projBrowser( value ):
    
    global projection
    global proj_img
    
    projection.grid_forget()
    
    proj_img = PIL.ImageTk.PhotoImage( image = PIL.Image.fromarray( np.uint8( (
            proj[ :, int( value ) - 1, : ] - np.amin( proj ) ) / ( np.amax(
                    proj ) - np.amin( proj ) ) * 255 ) ).resize( ( 640,
            np.ceil( detRows / detCols * 640 ).astype( int ) ) ) )
    
    projection = tkinter.Label( root, image = proj_img )
    projection.grid( row = 0 )

def recBrowser( value ):
    
    global recSlice
    global rec_img
    
    recSlice.grid_forget()
    
    rec_img = PIL.ImageTk.PhotoImage(
            image = PIL.Image.fromarray( np.uint8( rec[ int( value ) - 1, ... ]
            / np.amax( rec[ int( value ) - 1, ... ] ) * 255 ) ).resize( ( 640,
                     640 ) ) )
    
    recSlice = tkinter.Label( viewer, image = rec_img )
    recSlice.grid( row = 0 )

projSlider = tkinter.Scale( root, from_ = 1, to = num_angles, orient =
                           tkinter.HORIZONTAL, length = 640,  command =
                           projBrowser )
projSlider.grid( row = 1 )

def parameters():
    
    global parameters
    global start
    global end
    global allProj
    global projCheck
    global label20, label25
    global startSkip
    global endSkip
    global top
    global bottom
    global manualSet
    global autoCheck
    global num_iterations
    
    parameters = tkinter.Toplevel()
    parameters.title( 'DIRECTT - Reconstruction parameters' )
    parameters.geometry( '409x189' )
    
    tkinter.Label(
            parameters, text = 'Angular range: ', font = ( 'Verdana', 11 )
            ).grid( row = 0, column = 0 )
    tkinter.Label( parameters, text = ' to ', font = ( 'Verdana', 11 ) ).grid(
            row = 0, column = 2 )
    
    start = tkinter.Entry( parameters, width = 4, font = ( 'Verdana', 11 ) )
    start.insert( 0, 0 )
    start.grid( row = 0, column = 1 )
    
    end = tkinter.Entry( parameters, width = 4, font = ( 'Verdana', 11 ) )
    end.insert( 1, 180 )
    end.grid( row = 0, column = 3 )
    
    allProj = tkinter.IntVar()
    
    projCheck = tkinter.Checkbutton( parameters, text = 'Use all projections',
                                  variable = allProj, font = ( 'Verdana', 11 ),
                                  command = allProjections )
    projCheck.select()
    projCheck.grid( row = 1, columnspan = 4, sticky = tkinter.W )
    
    label20 = tkinter.Label( parameters, font = ( 'Verdana', 11 ) )
    label20.grid( row = 2, columnspan = 4 )
    label25 = tkinter.Label( parameters )
    label25.grid( row = 2, column = 4 )
    
    startSkip = tkinter.Label( parameters )
    startSkip.grid( row = 2, column = 4 )
    
    endSkip = tkinter.Label( parameters )
    endSkip.grid( row = 2, column = 6 )
    
    tkinter.Label( parameters, text = 'Reconstruct slices: ', font = (
            'Verdana', 11 ) ).grid( row = 3, column = 0, sticky = tkinter.E )
    tkinter.Label( parameters, text = ' to ', font = ( 'Verdana', 11 ) ).grid(
            row = 3, column = 2 )
    
    top = tkinter.Entry( parameters, width = 4, font = ( 'Verdana', 11 ) )
    top.insert( 0, 1 )
    top.grid( row = 3, column = 1 )
    
    bottom = tkinter.Entry( parameters, width = 4, font = ( 'Verdana', 11 ) )
    bottom.insert( 0, detRows )
    bottom.grid( row = 3, column = 3 )
    
    manualSet = tkinter.IntVar()
    
    autoCheck = tkinter.Checkbutton(
            parameters, text = 'Set number of iterations manually:', variable =
            manualSet, font = ( 'Verdana', 11 ), command = autoIterations )
    autoCheck.grid( row = 4, columnspan = 4, sticky = tkinter.W )
    
    num_iterations = tkinter.Entry(
            parameters, width = 4, font = ( 'Verdana', 11 ), state =
            tkinter.DISABLED )
    num_iterations.grid( row = 4, column = 4 )
    
    tkinter.Button( parameters, text = 'Reconstruct', font = ( 'Verdana', 11 ),
                                  command = reconstruct
                   ).grid( row = 5, columnspan = 5 )

def allProjections():
    
    global label20, label25
    global startSkip
    global endSkip
    
    label20.grid_forget()
    label25.grid_forget()
    startSkip.grid_forget()
    endSkip.grid_forget()
    
    if allProj.get():
        
        label20 = tkinter.Label( parameters, font = ( 'Verdana', 11 ) )
        label20.grid( row = 2, columnspan = 4 )
        label25 = tkinter.Label( parameters )
        label25.grid( row = 2, column = 4 )
        
        startSkip = tkinter.Label( parameters )
        startSkip.grid( row = 2, column = 4 )
        
        endSkip = tkinter.Label( parameters )
        endSkip.grid( row = 2, column = 6 )
    
    else:
        
        label20 = tkinter.Label( parameters, text = 'Do not use projections: ',
                                font = ( 'Verdana', 11 ) )
        label20.grid( row = 2, columnspan = 4, sticky = tkinter.E )
        label25 = tkinter.Label( parameters, text = ' to ', font = ( 'Verdana',
                                                                    11 ) )
        label25.grid( row = 2, column = 5 )
        
        startSkip = tkinter.Entry( parameters, width = 4, font = ( 'Verdana',
                                                                  11 ) )
        startSkip.insert( 0, 1 )
        startSkip.grid( row = 2, column = 4 )
        
        endSkip = tkinter.Entry( parameters, width = 4, font = ( 'Verdana', 11
                                                                ) )
        endSkip.insert( 0, 1 )
        endSkip.grid( row = 2, column = 6 )

def autoIterations():
    
    global num_iterations
    
    if manualSet.get():
        
        num_iterations = tkinter.Entry( parameters, width = 4, font = (
                'Verdana', 11 ) )
        num_iterations.insert( 0, 70 )
        num_iterations.grid( row = 4, column = 4 )
    
    else:
        
        num_iterations = tkinter.Entry( parameters, width = 4, font = (
                'Verdana', 11 ), state = tkinter.DISABLED )
        num_iterations.grid( row = 4, column = 4 )

def reconstruct():
    
    global proj
    global num_angles
    global rec
    global viewer
    global rec_img
    global recSlice
    
    print( 'Reconstructing ...' )
    
    theta = np.linspace( np.radians( float( start.get() ) ), np.radians( float(
            end.get() ) ), num_angles, False )
    
    if allProj.get() == 0:
        
        proj = np.append( np.append( proj[ :, :( int( startSkip.get() ) - 1 ),
                                          : ], proj[ :, int( endSkip.get() ):,
                                            : ], axis = 1 ), proj[ :, ( int(
                                                    startSkip.get() ) - 1
                                                    ):int( endSkip.get() ), :
                                                ], axis = 1 )
        
        theta = np.delete( theta, np.s_[ ( int( startSkip.get() ) - 1 ):int(
                endSkip.get() ) ] )
        
        num_angles = len( theta )
    
    proj2D = np.sum( proj[ ..., :num_angles, : ], axis = 0 )
    
    centreOfMass = np.zeros( num_angles )
    
    for i in range( num_angles ):
        
        centreOfMass[ i ] = scipy.ndimage.center_of_mass( proj2D[ i, : ] )[ 0 ]
    
    uOffset = ( detCols - 1 ) / 2 - scipy.optimize.curve_fit( curveFit, theta, 
          centreOfMass )[ 0 ][ 2 ]
    
    vectors = np.zeros( [ num_angles, 12 ] )
    vectors[ :, 0 ] = np.sin( theta )
    vectors[ :, 1 ] = - np.cos( theta )
    vectors[ :, 3 ] = np.cos( theta ) * uOffset
    vectors[ :, 4 ] = np.sin( theta ) * uOffset
    vectors[ :, 6 ] = np.cos( theta )
    vectors[ :, 7 ] = np.sin( theta )
    vectors[ :, 11 ] = 1
    
    if manualSet.get():
        
        rec = manual( vectors )
    
    else:
        
        rec = auto( vectors )
    
    if allProj.get() == 0:
        
        proj = np.append( np.append( proj[ :, :( int( startSkip.get() ) - 1 ),
                             : ], proj[ :, num_angles:, : ], axis = 1 ),
                        proj[ :, ( int( startSkip.get()
                             ) - 1 ):num_angles, : ], axis = 1 )
    
    parameters.destroy()
    
    viewer = tkinter.Toplevel()
    viewer.title( 'DIRECTT - Reconstruction of ' + root.filename )
    viewer.geometry( '645x730' )
    
    rec_img = PIL.ImageTk.PhotoImage(
            image = PIL.Image.fromarray( np.uint8( rec[ 0, ... ] / np.amax(
                    rec[ 0, ... ] ) * 255 ) ).resize( ( 640, 640 ) ) )
    
    recSlice = tkinter.Label( viewer, image = rec_img )
    recSlice.grid( row = 0 )
    
    recSlider = tkinter.Scale( viewer, from_ = 1, to = rec.shape[ 0 ], orient =
                           tkinter.HORIZONTAL, length = 640, command =
                           recBrowser )
    recSlider.grid( row = 1 )
    
    tkinter.Button( viewer, text = 'Save volume', font = ( 'Verdana', 11 ),
                                  command = save ).grid( row =
                  2 )

def curveFit( x, a, b, c ):
    
    return a * np.sin( x + b ) + c

def manual( v ):
    
    residual = np.copy( proj[ ( int( top.get() ) - 1 ):int( bottom.get() ),
                             :num_angles, : ] )
    
    detRows = residual.shape[ 0 ]
    
    projWeightAxis = np.sum( residual.reshape( detRows, num_angles * detCols ),
                            axis = 1 )
    
    vol_geom = astra.create_vol_geom( detCols, detCols, detRows )
    
    proj_geom = astra.create_proj_geom( 'parallel3d_vec', detRows, detCols, v )
    
    rec = np.zeros( [ detRows, detCols, detCols ], dtype = np.float32 )
    
    iteration = 0
    
    n = int( num_iterations.get() )
    
    while iteration < n:
        
        iteration += 1
        
        print( 'Iteration ', iteration, ' of ', n )
        
        bp_id = astra.data3d.create( '-vol', vol_geom )
        
        proj_id = astra.data3d.create( '-sino', proj_geom, residual )
        
        cfg = astra.astra_dict( 'BP3D_CUDA' )
        cfg[ 'ReconstructionDataId' ] = bp_id
        cfg[ 'ProjectionDataId' ] = proj_id
        
        alg_id = astra.algorithm.create( cfg )
        
        astra.algorithm.run( alg_id )
        
        bp = astra.data3d.get( bp_id )
        
        if iteration == 1:
            
            centerMass = np.rint(
                    scipy.ndimage.center_of_mass( np.sum( bp, axis = 0 ) )
                    ).astype( np.int )
            
            bpReshaped = bp.reshape( detRows, detCols ** 2 )
            
            B = ( np.amax( bpReshaped, axis = 1 ) - np.amax( bpReshaped * (
                    bpReshaped < bp[ :, centerMass[ 0 ], centerMass[ 1 ]
                    ].reshape( detRows, 1 ) ), axis = 1 ) )
            
            R = np.amax( bpReshaped, axis = 1 ).reshape( detRows, 1, 1 ) / (
                    num_angles * detCols )
            
            del bpReshaped
        
        else:
            
            R = np.amax( bp.reshape( detRows, detCols ** 2 ), axis = 1
                        ).reshape( detRows, 1, 1 ) / ( num_angles * detCols )
        
        bp -= np.amax( bp.reshape( detRows, detCols ** 2 ), axis = 1 ).reshape(
                detRows, 1, 1 ) - ( np.sum( residual.reshape( detRows,
                              num_angles * detCols ), axis = 1 ) /
                              projWeightAxis * B ).reshape( detRows, 1, 1 )
        bp *= bp > 0
        
        rec += bp / np.amax( bp.reshape( detRows, detCols ** 2 ), axis = 1
                            ).reshape( detRows, 1, 1 ) * R
        
        fp_id, fp = astra.create_sino3d_gpu( rec, proj_geom, vol_geom )
        
        residual = proj[ ( int( top.get() ) - 1 ):int( bottom.get() ),
                        :num_angles, : ] - fp
        
        astra.functions.clear()
    
    return( rec )

def auto( v ):
    
    residual = np.copy( proj[ ( int( top.get() ) - 1 ):int( bottom.get() ),
                             :num_angles, : ] )
    
    detRows = residual.shape[ 0 ]
    
    projWeight = np.sum( residual )
    
    projWeightAxis = np.sum( residual.reshape( detRows, num_angles * detCols ),
                            axis = 1 )
    
    vol_geom = astra.create_vol_geom( detCols, detCols, detRows )
    
    proj_geom = astra.create_proj_geom( 'parallel3d_vec', detRows, detCols, v )
    
    rec = np.zeros( [ detRows, detCols, detCols ], dtype = np.float32 )
    
    firstIteration = True
    
    while 1:
        
        bp_id = astra.data3d.create( '-vol', vol_geom )
        
        proj_id = astra.data3d.create( '-sino', proj_geom, residual )
        
        cfg = astra.astra_dict( 'BP3D_CUDA' )
        cfg[ 'ReconstructionDataId' ] = bp_id
        cfg[ 'ProjectionDataId' ] = proj_id
        
        alg_id = astra.algorithm.create( cfg )
        
        astra.algorithm.run( alg_id )
        
        bp = astra.data3d.get( bp_id )
        
        if firstIteration:
            
            centerMass = np.rint(
                    scipy.ndimage.center_of_mass( np.sum( bp, axis = 0 ) )
                    ).astype( np.int )
            
            bpReshaped = bp.reshape( detRows, detCols ** 2 )
            
            B = ( np.amax( bpReshaped, axis = 1 ) - np.amax( bpReshaped * (
                    bpReshaped < bp[ :, centerMass[ 0 ], centerMass[ 1 ]
                    ].reshape( detRows, 1 ) ), axis = 1 ) )
            
            R = np.amax( bpReshaped, axis = 1 ).reshape( detRows, 1, 1 ) / (
                    num_angles * detCols )
            
            del bpReshaped
            
            firstIteration = False
        
        else:
            
            R = np.amax( bp.reshape( detRows, detCols ** 2 ), axis = 1
                        ).reshape( detRows, 1, 1 ) / ( num_angles * detCols )
        
        bp -= np.amax( bp.reshape( detRows, detCols ** 2 ), axis = 1 ).reshape(
                detRows, 1, 1 ) - ( np.sum( residual.reshape( detRows,
                              num_angles * detCols ), axis = 1 ) /
                              projWeightAxis * B ).reshape( detRows, 1, 1 )
        bp *= bp > 0
        
        rec += bp / np.amax( bp.reshape( detRows, detCols ** 2 ), axis = 1
                            ).reshape( detRows, 1, 1 ) * R
        
        fp_id, fp = astra.create_sino3d_gpu( rec, proj_geom, vol_geom )
        
        residual = proj[ ( int( top.get() ) - 1 ):int( bottom.get() ),
                        :num_angles, : ] - fp
        
        residual2D = np.sum( residual.transpose( 1, 0, 2 ).reshape( num_angles,
                            detRows * detCols ), axis = 1 )
        
        if np.any( residual2D <= 0 ):
            
            print( '100  %  completed' )
            
            break
        
        else:
            
            print( '{:.1f}'.format( ( np.sum( fp ) / projWeight ) * 100 ),
                  '%  completed' )
        
        astra.functions.clear() 
    
    return( rec )

def save():
    
    root.filename = tkinter.filedialog.asksaveasfilename( initialdir = 'C:',
                                                         title = 'Save as' )
    
    imageio.volwrite( root.filename, rec )
    
    print( 'Volume saved')

tkinter.Button( root, text = 'Set reconstruction parameters', font = (
        'Verdana', 11 ), command = parameters ).grid( row = 2 )

root.mainloop()
