from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import sys
import os
import tensorflow as tf
print(tf.__version__)

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','common','dictionaries','grid.txt')

# ... (keep all the imports as they were)

def predict(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    print("\nLoading data from disk...")
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    print(f"Video object created. Face predictor path: {FACE_PREDICTOR_PATH}")
    try:
        if os.path.isfile(video_path):
            print(f"Loading video file: {video_path}")
            video.from_video(video_path)
        else:
            print(f"Loading frames from directory: {video_path}")
            video.from_frames(video_path)
        print("Data loaded successfully.")
        if not hasattr(video, 'data') or video.data is None:
            raise AttributeError("Video data not properly set")
        
        print(f"Video data shape: {video.data.shape}")
    except Exception as e:
        print(f"Error loading video: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

    # ... rest of the function remains the same

    print("Preparing data for prediction...")
    try:
        if K.image_data_format() == 'channels_first':
            img_c, frames_n, img_w, img_h = video.data.shape
        else:
            frames_n, img_w, img_h, img_c = video.data.shape
        print(f"Video shape: {video.data.shape}")
    except Exception as e:
        print(f"Error processing video data: {e}")
        return None, None
    
    print("Initializing LipNet model...")
    try:
        lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                        absolute_max_string_len=absolute_max_string_len, output_size=output_size)
        adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
        lipnet.load_weights(weight_path)
        print("Model initialized and weights loaded.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None

    print("Setting up decoder and spell checker...")
    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    
    print("Preparing input data for prediction...")
    X_data = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])
    
    # Apply the transposition here
    X_data = np.transpose(X_data, (0, 1, 3, 2, 4))
    
    print(f"Input shape: {X_data.shape}")
    print(f"Expected input shape: {lipnet.model.input_shape}")

    print("Running prediction...")
    try:
        # Ensure X_data is the correct shape
        if len(X_data.shape) == 4:  # If it's missing the batch dimension
            X_data = np.expand_dims(X_data, axis=0)
        
        print(f"X_data shape before prediction: {X_data.shape}")
        
        y_pred = lipnet.predict(X_data)
        print(f"Prediction shape: {y_pred.shape}")
        
        result = decoder.decode(y_pred, input_length)[0]
        print("Prediction completed successfully.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    return (video, result)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python predict.py [weight_path] [video_path] [optional: absolute_max_string_len] [optional: output_size]")
        sys.exit(1)

    weight_path = sys.argv[1]
    video_path = sys.argv[2]
    absolute_max_string_len = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    output_size = int(sys.argv[4]) if len(sys.argv) > 4 else 28

    video, result = predict(weight_path, video_path, absolute_max_string_len, output_size)

    if video is not None and result is not None:
        show_video_subtitle(video.face, result)

        stripe = "-" * len(result)
        print("")
        print("             --{}- ".format(stripe))
        print("[ DECODED ] |> {} |".format(result))
        print("             --{}- ".format(stripe))
    else:
        print("Prediction failed. Please check the error messages above.")

    print("Script execution completed.")
    
