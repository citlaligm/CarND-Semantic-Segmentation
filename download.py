# import zipfile
# import urllib2


# def download(directory):
#  webfile = urllib2.urlopen('http://kitti.is.tue.mpg.de/kitti/data_road.zip')
#  webfile2 = zipfile.ZipFile(webfile)
#  content = zipfile.ZipFile.open(webfile2).read()
#  localfile = open(directory, 'w')
#  localfile.write(content)
#  localfile.close()
#  return()

# download('./download')
url = 'http://kitti.is.tue.mpg.de/kitti/data_road.zip'

import requests, zipfile, io

r = requests.get(url)
print("Request done...")
z = zipfile.ZipFile(io.BytesIO(r.content))
print("Extracting...")
z.extractall()
print("Done")