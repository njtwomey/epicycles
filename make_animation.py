import numpy as np 

from svgpathtools import svg2paths, wsvg
from svgpathtools.path import Line, CubicBezier

import matplotlib.pyplot as pl 
from matplotlib.animation import FuncAnimation
import seaborn as sns 

from shutil import rmtree
from os.path import join, exists
from os import makedirs

sns.set_context('paper') 
sns.set_style('darkgrid') 

def load_svg(fname, rate=5): 
    xes = []

    paths, attributes = svg2paths(fname)
    vals = np.linspace(0, 1, 20)
    
    ind = 0
    print(len(paths))
    
    for path in paths[ind]: 
        start, end = path.start, path.end
        
        if isinstance(path, Line):
            if start.real < end.real: 
                start, end = end, start
            length = np.absolute(end - start)
            alpha = np.linspace(0, 1, int(np.ceil(length) / (3 * rate)), endpoint=False)
            if len(alpha):
                xes.extend(alpha * start + (1 - alpha) * end)
        elif isinstance(path, CubicBezier): 
            interp = path.poly()
            points = interp(vals)
            points = np.c_[points.real, points.imag]
            length = points[1:] - points[:-1]
            length = int(np.sqrt((length * length).sum()))
            rr = np.linspace(0, 1, int(length / rate) + 1, endpoint=False)
            xes.extend(interp(rr))
        else: 
            raise ValueError(f'{path} is an unrecognised type ({type(path)})')
    
    xx = np.asarray(xes)
    xx -= xx.mean(0)
    
    scale = max(np.abs(xx.real).max(), np.abs(xx.imag).max())
    xx /= scale
    
    if fname == 'cat2': 
        xx = xx[:-2]
    elif fname == 'dino-tri': 
        xx = xx[:-10]
    
    xx = xx.real - 1j * xx.imag
    
    pl.figure(figsize=(10, 10))
    pl.plot(xx.real, xx.imag, lw=0.5)
    pl.scatter(xx.real, xx.imag)

    return xx

def transform(f, threshold=0.9):
    # Calculate the FFT of the data
    F    = np.fft.fft(f) / f.size
    freq = np.fft.fftfreq(F.size, 1 / F.size)
    M = np.abs(F)

    # Select
    inds = np.argsort(-M)
    M_norm = M / M.sum()
    M_norm = np.cumsum(M_norm[inds])
    N = (M_norm < threshold).sum() + 1

    print(f'Evaluating with {N - 1} components')

    # Get the indexes of the top freqencies and slice these
    top_inds = (inds[:N])
    top_inds = top_inds[top_inds != 0]

    F_sel = F[top_inds]
    M_sel = M[top_inds]
    freq_sel = freq[top_inds]
    
    return N, F[0], F_sel, M_sel, freq_sel


def plot_circle(c, r, t, r_i, N=101, colour=None): 
    theta = np.linspace(-np.pi, np.pi, N)
    xy = np.c_[np.cos(theta) * r + c.real, np.sin(theta) * r + c.imag] 
    pl.plot(xy[:, 0], xy[:, 1], c=colour, lw=0.5)
    pl.plot([c.real, c.real + r_i.real], [c.imag, c.imag + r_i.imag], c=colour, lw=0.5)
    
def viz_factory(N, F_0, F_sel, M_sel, freq_sel, f): 
    points = []
    cmap = sns.color_palette(n_colors=N)
    def viz_func(ti_t): 
        print(ti_t)
        ti, t = ti_t
        v = F_0
        pl.clf()
        pl.plot(f.real, f.imag, 'k', lw=0.5)
        for ci, (F_i, M_i, f_i) in enumerate(zip(F_sel, M_sel, freq_sel)): 
            r_i = F_i * np.exp(1j * f_i * t) 
            pl.scatter([v.real], [v.imag], c=cmap[ci], lw=0.25)
            plot_circle(v, M_i, t, r_i, colour=cmap[ci])
            v += r_i
        points.append(v)
        pl.scatter([v.real], [v.imag], c='k')
        pp = np.asarray(points)
        pl.plot(pp.real, pp.imag, 'k', lw=1)
        pl.xticks([])
        pl.yticks([])
        pl.xlim((-2, 2))
        pl.ylim((-2, 2))
        pl.tight_layout()
        pl.savefig(f'out/{ti:05d}.png')
    return viz_func
    
def animate_file(fname, interval=100, n_frames=100, threshold=0.9, rate=1):
    # Load the data from file 
    f = load_svg(fname, rate=rate)

    # Do the spectral stuff
    N, F_0, F_sel, M_sel, freq_sel = transform(f, threshold=threshold)

    # Make directory structure
    dir_name = join('out')
    if exists(dir_name): 
        rmtree(dir_name)
    makedirs(dir_name)

    # Figure
    fig, ax = pl.subplots(1, 1, figsize=(10, 10))

    # Animation 
    viz_func = viz_factory(N, F_0, F_sel, M_sel, freq_sel, f)
    anim = FuncAnimation(
        fig=fig, 
        func=viz_func, 
        frames=list(enumerate(np.linspace(0, 2 * np.pi, n_frames))), 
        interval=interval,
    )

    anim.save(f'{fname}-{int(100 * threshold)}-{N}.gif', writer='imagemagick')
    
    
if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser(description='Approximate complex functions with Fourier decomposition.')
    parser.add_argument('--precision', dest='precision', default=0.99, help='The percentage precision in estimating the Fourier transform. In range (0-1). Default 0.99.', type=float)
    parser.add_argument('--nframes', dest='n_frames', default=100, help='The number of frames in the produced GIF. Default=100', type=int)
    parser.add_argument('--time', dest='time', default=10, help='The duration (in seconds) of the produced GIF. Default=10', type=int)
    parser.add_argument('--rate', dest='rate', default=2, help='Downsampling rate for the input image. Higher values give coarser approximation but will likely be faster to complete. Default=2.', type=float)
    parser.add_argument('svg', help='The name of the SVG file to be processed.')
    parser.print_help()

    args = parser.parse_args()

    interval = 1000 / (args.n_frames / args.time)

    animate_file(
        fname=args.svg, 
        interval=interval, 
        n_frames=args.n_frames, 
        threshold=args.precision, 
        rate=args.rate,
    )
