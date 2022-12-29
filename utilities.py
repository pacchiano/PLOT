import pickle
import zipfile
import os


def pickle_and_zip(obj, results_filename_stub, base_data_dir, is_zip_file = False):

  pickle_results_filename = "{}.p".format(results_filename_stub)
  ### start by saving the file using pickle

  pickle.dump( obj, 
    open("{}/{}".format(base_data_dir, pickle_results_filename), "wb"))

  if is_zip_file:

    zip_results_filename = "{}.zip".format(results_filename_stub)
    zip_file = zipfile.ZipFile("{}/{}".format(base_data_dir, zip_results_filename), 'w')

    zip_file.write("{}/{}".format(base_data_dir, pickle_results_filename), compress_type = zipfile.ZIP_DEFLATED, 
      arcname = os.path.basename("{}/{}".format(base_data_dir, pickle_results_filename)) )
    
    zip_file.close()

    os.remove("{}/{}".format(base_data_dir, pickle_results_filename))


def unzip_and_load_pickle(base_data_dir, results_filename_stub, is_zip_file = False):
  

  pickle_results_filename = "{}.p".format(results_filename_stub)

  ## If it is a ZIP file extract the pickle file.
  if is_zip_file:
    zip_results_filename = "{}.zip".format(results_filename_stub)
    
    
    zip_file = zipfile.ZipFile("{}/{}".format(base_data_dir, zip_results_filename), "r")
    zip_file.extractall(base_data_dir)

  results_dictionary = pickle.load( open("{}/{}".format(base_data_dir, pickle_results_filename), "rb") )

  ## If it is a ZIP file, delete the pickle file.
  if is_zip_file:
    os.remove("{}/{}".format(base_data_dir, pickle_results_filename))

  return results_dictionary




# def pickle_and_zip(obj, filename):
#     # Pickle the object
#     with open(filename, 'wb') as f:
#         pickle.dump(obj, f)
        
#     # Compress the file using zip
#     with zipfile.ZipFile(filename+'.zip', 'w', zipfile.ZIP_DEFLATED) as zip_f:
#         zip_f.write(filename)
        
#     # Delete the unpickled file
#     os.remove(filename)



# def unzip_and_load_pickle(filename):
#   zip_file_path = filename+'.zip'
#   pickle_file_name = filename+'.p'
#   # Unzip the file
#   with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall('.')

#   # Load the pickle file
#   with open(pickle_file_name, 'rb') as f:
#     data = pickle.load(f)

#   # Delete the zip file and the extracted pickle file
#   os.remove(zip_file_path)
#   os.remove(pickle_file_name)

#   return data
#     