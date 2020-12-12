import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from scipy.signal import unit_impulse

def b2r(img):
  """Returns BGR `img` in RGB."""
  return cv.cvtColor(img,cv.COLOR_BGR2RGB)
def r2h(img):
  """Returns RGB `img` in HSV."""
  return cv.cvtColor(img,cv.COLOR_RGB2HSV)
def h2r(img):
  """Returns HSV `img` in RGB."""
  return cv.cvtColor(img,cv.COLOR_HSV2RGB)
def rect(x,w,y,h,center=None):
  if center:
    return (slice(y-h//2,y+h-h//2),slice(x-w//2,x+w-w//2))
  return (slice(y,y+h),slice(x,x+w))
def circle(shape, radius=None, center=None):
  img_w = shape[0]; img_h = shape[1]
  if center is None: # use the middle of the image
      center = [int(img_w/2), int(img_h/2)]
  if radius is None: # use the smallest distance between the center and image walls
      radius = min(center[0], center[1], img_w-center[0], img_h-center[1])
  Y, X = np.ogrid[:img_h, :img_w]
  dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
  mask = dist_from_center <= radius
  return mask
def merge_masks(shape,*args):
  img_w = shape[0]; img_h = shape[1]
  ret = np.zeros((img_h,img_w),dtype=bool)
  for mask in args:
    ret[mask] = 1
  return ret
def kernelLaplace():
  return np.array([[0,1,0],[1,4,1],[0,1,0]])
def kernelProm(size):
  return np.array((1/(size**2))*np.ones((size,size)))
def kernelGauss(size,sigma=None):
  if (sigma == None or sigma<=0):
    sigma = 0.3*((size-1)*0.5 - 1) + 0.8
  x = y = np.arange(size) - int(np.floor(size/2))
  ret = np.zeros((size,size),dtype=float)
  for i in range(size):
    for j in range(size):
      ret[i,j] = np.exp(-(x[i]**2 + y[j]**2)/(2*sigma**2))
  ret /= np.sum(ret)
  return ret
def normalize(img,max_=255.0):
  """Normalizes `img` between 0 and `max_` (default 255)."""
  img -= img.min()
  img = (img*max_/img.max()).astype('uint8')
  return img
def lut(array):
  array = normalize(array)
  array = np.clip(array, 0, 255).astype('uint8')
  return array
def expandlut(min,max):
  array = np.array([
    (255*x/(max-min) - 255*min/(max-min)) if (x>min and x<max)
    else 0 if x<=min
    else 255
    for x in range(256)])
  array = normalize(array)
  array = np.clip(array, 0, 255).astype('uint8')
  return array
def loglut():
  log = np.log(1+np.arange(0,256))
  log = normalize(log)
  # log = np.clip(log, 0, 255).astype('uint8')
  return log
def powlut(gamma):
  ppow = np.arange(0,256)**gamma
  ppow = normalize(ppow)
  # ppow = np.clip(ppow, 0, 255).astype('uint8')
  return ppow
def prom(*args):
  if type(args[0]) is list:
    ret = np.zeros_like(args[0][0],dtype='float64')
    for arg in args[0]:
      ret += arg
    return (ret/len(args[0])).astype(args[0][0].dtype)
  else:
    ret = np.zeros_like(args[0],dtype='float64')
    for arg in args:
      ret += arg
    return (ret/len(args)).astype(args[0].dtype)
def mult(img,mask):
  return img*mask
def highboost(img,A,ksize=3,hue=False):
  kernelHB = A*unit_impulse((ksize,ksize),'mid') - kernelGauss(ksize,-1)
  if hue: return (cv.filter2D(img,cv.CV_16S,kernelHB)).astype('uint8')
  return normalize(cv.filter2D(img,cv.CV_16S,kernelHB),255.).astype('uint8')
def equalize(img):
  """Returns an equalized version of single channel `img`."""
  hist = cv.calcHist([img],[0],None,[256],[0,256])
  H = hist.cumsum()
  H = H * hist.max()/ H.max()
  lin = H*255/max(H)
  lut = np.clip(lin, 0, 255)
  lut = lut.astype('uint8')
  return lut[img]
def equalizergb(img):
  """Returns a RGB-channels equalized version of `img`."""
  r,g,b = cv.split(img)
  r=equalize(r)
  g=equalize(g)
  b=equalize(b)
  return cv.merge([r,g,b])
def equalizev(img):
  """Returns a v-channel equalized version of `img`."""
  h,s,v = cv.split(cv.cvtColor(img,cv.COLOR_RGB2HSV))
  v=equalize(v)
  return cv.cvtColor(cv.merge([h,s,v]),cv.COLOR_HSV2RGB)
def mse(A,B,axis=None):
  return np.square(np.subtract(A, B)).mean(axis=axis)
def fft(img,log=False,magnitude=False):
  IMG = np.fft.fftshift(np.fft.fft2(img.astype('float32')))
  if not magnitude: return IMG
  mg = cv.magnitude(IMG.real,IMG.imag)
  if not log:
    return cv.magnitude(IMG.real,IMG.imag)
  else:
    mg = np.log(mg+1)
    return cv.normalize(mg,mg,0,1,cv.NORM_MINMAX)
def ifft(IMG):
  return normalize(np.real(np.fft.ifft2(np.fft.ifftshift(IMG)))).astype('uint8')
def rotate(img, angle):
  """Rotates `img` by `angle` degrees around the center"""
  r = cv.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle, 1.0)
  return cv.warpAffine(img, r, img.shape)
def noised_gauss(img,std):
  """Returns a pair `[image,noise]`
  where `image` is `img` with added gaussian noise
  with `std` standard deviation
  and `noise` is the pattern added by that noise."""
  noise = np.random.normal(0,std,img.shape)
  img_noise = np.clip(img.astype(float)+noise,0,255).astype('uint8')
  return img_noise,noise
def noised_unif(img,min_,max_):
  """Returns a pair `[image,noise]`
  where `image` is `img` with added uniform noise
  with `min_` and `max_` values
  and `noise` is the pattern added by that noise."""
  noise = np.random.uniform(min_,max_,img.shape)
  img_noise = np.clip(img.astype(float)+noise,0,255).astype('uint8')
  return img_noise,noise
def noised_snp(img,pad):
  """Returns a pair `[image,noise]`
  where `image` is `img` with added salt-and-pepper noise
  and `pad` being a measure of its noisiness
  and `noise` is the pattern added by that noise."""
  noise = np.random.randint(0,255,img.shape)
  img_noise = img.copy()
  img_noise[noise < pad] = 0
  img_noise[noise > 255-pad] = 255
  noise[noise < pad] = 0
  noise[noise > 255-pad] = 255
  noise[(noise != 0) & (noise != 255)] = 127
  return img_noise,noise
def fill_holes(img,kernel):
  I = img//255
  Ic = 1-I

  F = np.zeros_like(I)
  F[:,0] = Ic[:,0]
  F[:,-1] = Ic[:,-1]
  F[0,:] = Ic[0,:]
  F[-1,:] = Ic[-1,:]

  # cv.namedWindow("F",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
  # cv.namedWindow("dif",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

  dif = np.zeros_like(img).astype(bool)
  while np.any(~dif):
    # print("loop")
    Fnew = cv.dilate(F,kernel)*Ic
    dif = F == Fnew
    # cv.imshow("F",F)
    # cv.imshow("dif",dif.astype('uint8'))
    # print(dif)
    # cv.waitKey(1)
    F = Fnew
  return (1-F)*255


### Drawing functions

def hist(img,ax=None,ref_ax=None,cdf=False,real=False,dpi=None):
  """Draw histogram of `img` in `ax`,
  with aspect ratio given by `ref_ax`
  (which should be the axes the image was drawn in).
  Set `cdf` to True to plot cumulative distribution function
  on top."""
  f = None
  if ax==None:
    f = plt.figure(dpi=dpi)
    ax = plt.gca()
  im = img.ravel()
  if not real:
    ax.hist(im,256,[0,256])
    ax.set_xlim((-10,265))
    ax.set_xticks([0,25,50,75,100,125,150,175,200,225,255])
  else:
    ax.hist(im,512)
  ax.tick_params(labelsize=5,pad=.01,width=.25,labelrotation=30)
  if ref_ax:
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    asp /= np.abs(np.diff(ref_ax.get_xlim())[0] / np.diff(ref_ax.get_ylim())[0])
    ax.set_aspect(asp)
  return f
  if cdf:
    ax2 = ax.twinx()
    hist,_ = np.histogram(im,256,[0,256])
    ax2.plot(np.cumsum(hist),'r--',alpha=0.7)
    ax2.tick_params(right=False,labelright=False,bottom=False,labelbottom=False)
    if ref_ax:
      ax2.set_aspect(asp)
  return f
def colhist(img,type:"None|joined|split"=None,dpi=None):
  """Draw `img` and all three channels' histograms in
  subplots. `type` can be:
    'joined': all three histograms in a single axes, default
    'split': three separate histograms"""
  r,g,b = (cv.split(img)); r = r.ravel(); g = g.ravel(); b = b.ravel()
  rc = (1,0,0,.5); gc = (0,1,0,.5); bc = (0,0,1,.5)
  f,a = plt.subplots(1,4 if type=='split' else 2,dpi=dpi)
  a[0].imshow(img); a[0].set_xticks([]); a[0].set_yticks([])

  if type == None or type == 'joined':
    # f.subplots_adjust(wspace=0.1,right=3,bottom=-.5)
    a[1].hist([r,g,b],256,[0,256],color=[rc,gc,bc],histtype='stepfilled')
    asp = np.diff(a[1].get_xlim())[0] / np.diff(a[1].get_ylim())[0]
    asp /= np.abs(np.diff(a[0].get_xlim())[0] / np.diff(a[0].get_ylim())[0])
    a[1].set_aspect(asp)
  elif type == 'split':
    # f.subplots_adjust(wspace=0.2,right=4)
    a[1].hist(r,256,[0,256],color='r')
    a[2].hist(g,256,[0,256],color='g')
    a[3].hist(b,256,[0,256],color='b')
    asp = np.diff(a[1].get_xlim())[0] / np.diff(a[1].get_ylim())[0]
    asp /= np.abs(np.diff(a[0].get_xlim())[0] / np.diff(a[0].get_ylim())[0])
    a[1].set_aspect(asp)
    a[2].set_aspect(asp)
    a[3].set_aspect(asp)
  return f
def lutshow(img,lut):
  """Draw `img` and a `lut` transformation,
  and the result of applying it to `img`"""
  f,ax = plt.subplots(1,3,dpi=150)
  imshow(img,ax[0])
  ax[1].plot(lut)
  ax[1].plot(np.arange(0,256),'--')
  ax[1].set_aspect('equal', 'box')
  ax[1].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
  imshow(lut[img],ax[2])
  return f
def imshow(img,ax=None,title=None,tsize=None,dpi=None,vmin=None,vmax=None,interactive=False):
  """Draw `img` in `ax` with `title` caption on top,
  of size `tsize`.

  For single channel images,
  `vmin` and `vmax` are set automatically,
  but you may set custom values to specify
  the range between which grays will be drawn
  (values outside of the range
  will be either black or white).
  """
  if not (img<=1).all() and (img>=0).all() and (img<=255).all():
    if vmin==None: vmin = 0
    if vmax==None: vmax = 255
  f = None
  if ax==None:
    f = plt.figure(dpi=dpi)
    ax = plt.gca()
  axImage = ax.imshow(img,vmin=vmin,vmax=vmax,cmap='gray',interpolation='none')
  ax.set_xticks([])
  ax.set_yticks([])
  if title:
    ax.set_title(title,dict(size=tsize))
  if interactive: return f, axImage
  else: return f
def channelplot(img,model:"rgb|hsv"="rgb",title="img",dpi=None):
  if model=="rgb":
    [t0,t1,t2] = "rgb"
    [c0,c1,c2] = cv.split(img)
  elif model=="hsv":
    [t0,t1,t2] = "hsv"
    [c0,c1,c2] = cv.split(cv.cvtColor(img,cv.COLOR_RGB2HSV))

  f,a = plt.subplots(1,4,dpi=dpi)

  imshow(img,a[0]); a[0].set_title(title)
  imshow(c0,a[1]); a[1].set_title(t0)
  imshow(c1,a[2]); a[2].set_title(t1)
  imshow(c2,a[3]); a[3].set_title(t2)
  return f
def fftshow(img,dpi=150,alpha=0.9,log=False,threed=False,interactive=False):
  """Plots `img` and its DFT magnitude in log scale
  in both 2D and 3D views.
  """
  if threed:
    f,a = plt.subplots(1,3,dpi=dpi)
    f.subplots_adjust(right=0.01,left=-0.4)
    IMG = fft(img,log=log,magnitude=True)
    imshow(img,a[0])
    a[0].axis('off')
    imshow(IMG,a[1])
    a[1].axis('off')
    a[2].remove()
    ax = f.add_subplot(1, 3, 3, projection='3d')
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    x = np.linspace(0,img.shape[1]-1,img.shape[1])
    y = np.linspace(0,img.shape[0]-1,img.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X,Y,IMG,cmap='gray',alpha=alpha, shade=False, lw=.5)
    ax.set_aspect('equal', 'box')
    return f
  else:
    f,a = plt.subplots(1,2,dpi=dpi)
    IMG = fft(img,log=log,magnitude=True)
    _,axImage1 = imshow(img,a[0],interactive=True)
    a[0].axis('off')
    _,axImage2 = imshow(IMG,a[1],interactive=True)
    a[1].axis('off')
    if interactive:
      return f, (axImage1,axImage2)
    else:
      return f
def save(f,path_or_page,dpi=None):
  """Saves figure `f` to `path`.

  Modifies `f`'s background alpha in place."""
  f.patch.set_alpha(0)
  if type(path_or_page) is str:
    f.savefig(path_or_page,dpi=dpi,bbox_inches="tight",transparent=True,interpolation='none',pad_inches=0,tight=True)
  else:
    path_or_page.savefig(f,dpi=dpi,bbox_inches="tight",transparent=True,interpolation='none')

