from tempfile import TemporaryFile
import numpy as np
outfile = TemporaryFile()
x = np.arange(10)
np.save(outfile, x)
x=np.load(outfile)
print(x)
