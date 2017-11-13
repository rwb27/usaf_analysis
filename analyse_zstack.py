data = np.load("edge_summary.npz")
psfs = data['psfs']
psf8 = (psfs/psfs.max()*255).astype(np.uint8)
psf8[psfs<0] = 0 # important - otherwise it all goes wrong when we plot

blocks = psfs.shape[1]

fig, axes = plt.subplots(1,blocks, figsize=(2,1*blocks)
for i in range(blocks):
    axes[i].imshow(psf8[:,i,...], aspect="auto")

    