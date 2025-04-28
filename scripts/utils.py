import os
import shutil

def set_path(path, verbose=False):
    if os.path.isdir(path):
        if verbose:
            print('use existed path: {}'.format(path))
    else:
        os.makedirs(path)
        if verbose:
            print('created path: {}'.format(path))

def empty_dir(folder):
  for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e)) 