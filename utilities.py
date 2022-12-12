import pickle
import zipfile
import os

def pickle_and_zip(obj, filename):
    # Pickle the object
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        
    # Compress the file using zip
    with zipfile.ZipFile(filename+'.zip', 'w', zipfile.ZIP_DEFLATED) as zip_f:
        zip_f.write(filename)
        
    # Delete the unpickled file
    os.remove(filename)
