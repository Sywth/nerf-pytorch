# START TBD : Quick Rendering Test for NeRF
angles = np.linspace(0, 2 * np.pi, num_scans)
imgs_shape = (len(angles), size, size)
imgs = np.zeros(imgs_shape)
for i, angle in enumerate(angles):
    # TODO figure out how to get this to sync with the existing astra scan and line up nicely. 
    # TODO figure out how to get this to 
    # TODO Check if radius changes the nerf model, as it shouldn't as its orthographic
    # TODO Figure out why background is white (1.0) when it should be black (0.0)
    imgs[i] = utils.rgb_to_mono(
        get_image_at_theta(-angle, hwf_rendering, render_kwargs_test)
    )

utils.create_gif(
    imgs,
    angles,
    "Angle {}",
    f"./figs/temp/ct_nerf_neg.gif",
)
# END TBD  Quick Rendering Test for NeRF
