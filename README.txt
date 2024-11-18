*********************************** WELCOME to the really cool and not under preforming model that is the TORGG-16 model ***********************************
This model was developed for our CSC 606 class at USM and aims to predict what type of tornado is present in a given photo of cloud formations 
    the dataset that we recommend if the tornet dataset from MIT/USA Airforce 
    which can be found here https://github.com/mit-ll/tornet (scroll down a little to "Downloading the Data" )
        * note you do not need to download the .csv for this model to work 



*********************************** CONVERTING THE .nc FILE TO IMAGES ***********************************
    To convert the .nc files you gathered from the tornet dataset (doesnt mater what year ) you will need to run the script "NC_to_Image.py" that is located here in the Scripts directory
    if you want to change the year you pull the information from, all you need to do is modify line # 103 "if category in subdir and '____' in subdir:  where '___' is whatever year of information you want to download 
    if you only want to download a percentage of the data from that year there is a comment that will help you with that

    when you run the script youre output will start filling up FAST, it should look something like this
    
    '
        Saved image: Tornet_Dataset_Images\Test\0001\TOR_131004_030848_KOAX_472673_A1.jpg
        [0 0 0 0]
        Saved image: Tornet_Dataset_Images\Test\0000\WRN_131006_022303_KHPX_1073337n_U7.jpg
        [0 0 0 0]
        Saved image: Tornet_Dataset_Images\Test\0000\WRN_131006_015602_KPAH_1073336n_F4.jpg
        [0 0 0 0]
        Saved image: Tornet_Dataset_Images\Test\0000\NUL_131006_200940_KILN_477705s_I6.jpg
        [0 0 0 0]
    '

    The [0 0 0 0] under each saved image is the class it is getting saved into (there are 5 of them)



*********************************** MODIFYING THE IMAGES CONVERETED ***********************************
    if you want to modify the image types / want to test the model with different image types 
    there are 6 different image types in the .nc files that can extracted, 'VEL' 'DBZ' 'KDP' 'RHOHV' 'ZDR' 'WIDTH' 
            to change the different image type, id reccoment just doing command f and searching for 'VEL', then just changing every instance of 'VEL' with your new variable you'll looking for
        each of these images display different data and will require changing the colormap that the image is processed with 
        for 'VEL' I'd reccoment 'bwr' or 'seismic' (really any of the Diverging Colormaps)

        if you want to look at all the different colormaps here is a link -> https://matplotlib.org/stable/gallery/color/colormap_reference.html
        depending on what image type you extract, you'll need to match a colormap to it.

        to modify all the images settings it can be found in the method process_nc_file towards the end of it 
        specifically in the plt.imshow(), youll have cmap = colormap.



*********************************** RUNNING THE MODEL ***********************************
    If you have a nvidia GPU the model should run much faster for you than it did for me (RIP AMD / ROCm... Keras support)
        IF YOU DONT LISTEN UP HERE!!!!
            *** you will need to uncomment the os.environ lines in the code as well as the p = / p.___ lines (lines 163-172) and modify the values so you dont blue screen your PC 
                depending on your CPU you will have to change values.
                    !!!IF YOU DO NOT UNCOMMENT THOSE LINES YOU WILL BLUESCREEN!!!!

    but you shouldnt really need to modify a ton (unless you are trying to train the model to its fullest) 

    however, you'll need to change a couple things depending on how you have your files setup (if you modified anything in the script) you just need to make sure that the variables
        train_dir = WHERE EVER YOUR TRAIN DATASET IS LOCATED (GIVE EITHER REL PATH OR DIRECT PATH)
        test_dir = WHERE EVER YOUR TEST DATASET IS LOCATED (GIVE EITHER REL PATH OR DIRECT PATH)

    at the current moment (11/16/2024) our model is severly overfitting and needs a lot of work before it can be reliable in predicting tornadoes in images.



*********************************** LOOING AT LOGS ***********************************
    for logging we are using tensorboard (highly recommend), it makes understanding what is going on much easier. 
    to use this in VSCode all you have to do is hit CONTROL + SHIFT + P then select "Python: Launch Tensorboard"
        once you do that you'll have a menu appear at the top and need to select "Select Other Folder" and choose where the logs are being stored, by default they should be stored here "logs\fit" 
            you can modify the log location to wherever you want them to go 

