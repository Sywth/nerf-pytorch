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



class AstraScanVec3D:
    def __init__(self, phantom_shape, euler_angles, img_res = 64.0):
        self.phantom_shape = phantom_shape
        self.euler_angles = np.array(euler_angles)  # shape: (num_projections, 3)
        self.inv_res = phantom_shape[0] / img_res
        
        # These will be created in re_init()
        self.vol_geom = None
        self.proj_geom = None
        self.detector_resolution = None
        
        self.re_init()
    
    def re_init(self):
        self.vol_geom = astra.create_vol_geom(*self.phantom_shape)
        # Detector resolution derived from the phantom shape (using first two dimensions)
        self.detector_resolution = (np.array(self.phantom_shape[:2]) / self.inv_res).astype(int)
        
        # Compute the 12-element projection vectors from Euler angles.
        vectors = self._compute_projection_vectors()
        self.proj_geom = astra.create_proj_geom(
            'parallel3d_vec',
            self.detector_resolution[0],  # number of detector rows
            self.detector_resolution[1],  # number of detector columns
            vectors
        )
    
    def _compute_projection_vectors(self):
        num_projections = len(self.euler_angles)
        vectors = np.zeros((num_projections, 12), dtype=float)
        for i, (theta, phi, gamma) in enumerate(self.euler_angles):
            R = self._euler_to_rotation_matrix(theta, phi, gamma)
            # Define base directions:
            base_ray = np.array([0, 0, 1])
            base_u = np.array([1, 0, 0])  # direction for detector column increase
            base_v = np.array([0, 1, 0])  # direction for detector row increase
            
            ray = R @ base_ray
            # Center of detector is set to origin (could be offset if needed)
            d = np.zeros(3)
            # Scale the detector pixel vectors by inv_res (serves as DetectorSpacing)
            u = R @ base_u * self.inv_res
            v = R @ base_v * self.inv_res
            
            # Pack the 12 elements: ray, detector center, u, and v
            vectors[i, 0:3] = ray
            vectors[i, 3:6] = d
            vectors[i, 6:9] = u
            vectors[i, 9:12] = v
        return vectors

    def _euler_to_rotation_matrix(self, theta, phi, gamma):
        return Rotation.from_euler("ZYX", [theta, phi, gamma], degrees=False).as_matrix()

    def get_ct_camera_poses(self, radius=2.0):
        """
        # TODO
        This is not workign inline with astra but im going assume its fine for most 360 spin cases 
        """
        num_projections = len(self.euler_angles)
        poses = np.zeros((num_projections, 4, 4), dtype=float)

        # Obtain the 12-element vectors for each projection from the astra geometry.
        vectors = astra.geom_2vec(self.proj_geom)['Vectors']

        # With reference to astra https://astra-toolbox.com/docs/geom3d.html#projection-geometries
            # ray : the ray direction
            # d : the center of the detector
            # u : the vector from detector pixel (0,0) to (0,1)
            # v : the vector from detector pixel (0,0) to (1,0)

        for i, (rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) in enumerate(vectors):
            # Extract and normalize the ray direction.
            ray = np.array([rayX, rayY, rayZ])
            ray /= np.linalg.norm(ray)

            # Use the 'u' vector to define the right direction (normalize it).
            right = np.array([uX, uY, uZ])
            right /= np.linalg.norm(right)

            # Compute the camera's up vector to form a right-handed coordinate system.
            # (Cross product of ray and right; note: order matters.)
            up = np.cross(ray, right)
            up /= np.linalg.norm(up)

            # In our convention the camera looks along -z, so we want:
            # R @ [0, 0, -1] = ray  =>  R's third column = -ray.
            R = np.column_stack((right, up, -ray))

            # Position the camera at -radius along the ray direction so that
            # the vector from the camera to the origin is aligned with ray.
            t = -radius * ray

            # Build the 4x4 pose matrix (camera-to-world).
            pose = np.eye(4, dtype=float)
            pose[:3, :3] = R
            pose[:3, 3] = t

            poses[i] = pose

        return poses

    def generate_ct_imgs(self, phantom):
        """
        Generate CT projection data (sinograms) from the given phantom.
        
        Parameters:
            phantom (np.ndarray): The 3D phantom (volume) data.
            
        Returns:
            np.ndarray: The sinogram data with axes rearranged appropriately.
            
        Raises:
            ValueError: If the projection or volume geometry has not been initialized.
        """
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError("Projection geometry and volume geometry must be initialized.")
        
        volume_id = astra.data3d.create("-vol", self.vol_geom, phantom)
        proj_id = astra.create_projector("cuda3d", self.proj_geom, self.vol_geom)
        sinogram_id, sinogram = astra.create_sino3d_gpu(volume_id, self.proj_geom, self.vol_geom, returnData=True)
        
        # Free resources
        astra.data3d.delete(volume_id)
        astra.projector.delete(proj_id)
        astra.data3d.delete(sinogram_id)
        
        # Rearranging axes if necessary (this step can be adjusted based on downstream use)
        return np.moveaxis(sinogram, 0, 1)
    
    def reconstruct_3d_volume_sirt(self, ct_imgs, num_iterations=64):
        """
        Reconstruct a 3D volume from the given sinogram data using the SIRT3D_CUDA algorithm.
        
        Parameters:
            sinogram (np.ndarray): The projection data.
            num_iterations (int): Number of iterations for the reconstruction algorithm.
            
        Returns:
            np.ndarray: The reconstructed 3D volume.
            
        Raises:
            ValueError: If the projection or volume geometry has not been initialized.
        """
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError("Projection geometry and volume geometry must be initialized.")
        
        sinogram = ct_imgs.swapaxes(0, 1)
        sinogram_id = astra.data3d.create("-proj3d", self.proj_geom, sinogram)
        reconstruction_id = astra.data3d.create("-vol", self.vol_geom)
        
        # Configure the reconstruction algorithm (here SIRT3D_CUDA is used)
        alg_cfg = astra.astra_dict("SIRT3D_CUDA")
        alg_cfg["ProjectionDataId"] = sinogram_id
        alg_cfg["ReconstructionDataId"] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)
        
        astra.algorithm.run(algorithm_id, num_iterations)
        reconstruction = astra.data3d.get(reconstruction_id)
        
        # Clean up
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(sinogram_id)
        astra.data3d.delete(reconstruction_id)
        
        return reconstruction
                
    def reconstruct_3d_volume_cgls(self, ct_imgs, num_iterations=64):
        """
        Reconstruct a 3D volume from the given sinogram data using the CGLS3D_CUDA algorithm.

        Parameters:
            sinogram (np.ndarray): The projection data (sinogram).
            num_iterations (int): Number of iterations for the CGLS algorithm.

        Returns:
            np.ndarray: The reconstructed 3D volume.

        Raises:
            ValueError: If the projection or volume geometry has not been initialized.
        """
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError("Projection geometry and volume geometry must be initialized.")

        sinogram = ct_imgs.swapaxes(0, 1)

        # Create ASTRA data objects for the sinogram and reconstruction volume
        sinogram_id = astra.data3d.create("-proj3d", self.proj_geom, sinogram)
        reconstruction_id = astra.data3d.create("-vol", self.vol_geom)

        # Configure the CGLS3D_CUDA reconstruction algorithm
        alg_cfg = astra.astra_dict("CGLS3D_CUDA")
        alg_cfg["ProjectionDataId"] = sinogram_id
        alg_cfg["ReconstructionDataId"] = reconstruction_id

        # Create and run the reconstruction algorithm
        algorithm_id = astra.algorithm.create(alg_cfg)
        astra.algorithm.run(algorithm_id, num_iterations)

        # Retrieve the reconstructed volume
        reconstruction = astra.data3d.get(reconstruction_id)

        # Clean up resources
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(sinogram_id)
        astra.data3d.delete(reconstruction_id)

        return reconstruction




# %%








































# %%


@dataclass
class AstraScanParameters:
    phantom_shape: tuple
    num_scans: int = 64
    inv_res: float = 1.0
    min_theta: float = 0
    max_theta: float = 2 * np.pi

    proj_geom: Any | None = None
    vol_geom: Any | None = None

    def re_init(self):
        """
        Reset using the current parameters
        """
        self.vol_geom = astra.create_vol_geom(*self.phantom_shape)
        self.detector_resolution = (
            np.array(self.phantom_shape[:2]) / self.inv_res
        ).astype(int)

        angles = self.get_angles_rad()
        self.proj_geom = astra.create_proj_geom(
            "parallel3d",
            self.inv_res,  # DetectorSpacingX
            self.inv_res,  # DetectorSpacingY
            self.detector_resolution[0],  # DetectorRowCount
            self.detector_resolution[1],  # DetectorColCount
            angles,  # ProjectionAngles
        )

    def __post_init__(self):
        self.re_init()

    def get_angles_rad(self):
        return np.linspace(
            self.min_theta, self.max_theta, self.num_scans, endpoint=False
        )

    def reconstruct_3d_volume_alg(self, sinogram, num_iterations=4):
        """
        Note this will only work for sinograms generated using this instance
        """
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError("proj_geom and vol_geom must be set before reconstruction")

        sinogram_id = astra.data3d.create("-proj3d", self.proj_geom, sinogram)
        reconstruction_id = astra.data3d.create("-vol", self.vol_geom)

        # Initialize algorithm parameters
        alg_cfg = astra.astra_dict("SIRT3D_CUDA")
        alg_cfg["ProjectionDataId"] = sinogram_id
        alg_cfg["ReconstructionDataId"] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)

        astra.algorithm.run(algorithm_id, num_iterations)
        reconstruction = astra.data3d.get(reconstruction_id)

        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(sinogram_id)
        astra.data3d.delete(reconstruction_id)

        return reconstruction

    def generate_ct_imgs(self, phantom):
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError(
                "proj_geom and vol_geom must be set before sinogram generation"
            )

        volume_id = astra.data3d.create("-vol", self.vol_geom, phantom)
        proj_id = astra.create_projector("cuda3d", self.proj_geom, self.vol_geom)
        sinogram_id, sinogram = astra.create_sino3d_gpu(
            volume_id, self.proj_geom, self.vol_geom, returnData=True
        )

        # Free GPU memory
        astra.data3d.delete(volume_id)
        astra.projector.delete(proj_id)
        astra.data3d.delete(sinogram_id)

        return np.moveaxis(sinogram, 0, 1)
def acquire_ct_data(
    phantom: np.ndarray,
    num_scans=16,
    use_rgb=True,
    normalize_instead_of_standardize=True,
):
    scan_params = AstraScanParameters(
        phantom_shape=phantom.shape,
        num_scans=num_scans,
    )
    ct_imgs = scan_params.generate_ct_imgs(phantom)
    ct_poses = utils.generate_camera_poses(
        angles=scan_params.get_angles_rad(), 
        radius=4.0,
        axis="x"
    )

    if use_rgb:
        ct_imgs = utils.mono_to_rgb(ct_imgs)

    if normalize_instead_of_standardize:
        ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())
    else:
        # we standardize the images
        ct_imgs = (ct_imgs - ct_imgs.mean()) / ct_imgs.std()

    # Remove background and cast to RGBA
    ct_imgs = rgb_to_rgba(ct_imgs)
    ct_imgs = remove_bg(ct_imgs)

    return ct_imgs, ct_poses, scan_params.get_angles_rad()

# %%

def old():
    ct_mode = True
    visible_mode = False

    phantom_idx = 13
    size = 320
    num_scans = 64
    use_rgb = True

    phantom = load_phantom(phantom_idx, size)
    if ct_mode:
        fov_deg = float("inf")
        ct_imgs, ct_poses, ct_angles = acquire_ct_data(phantom, num_scans, use_rgb)
        npz_dict = get_npz_dict(ct_imgs, ct_poses, ct_angles, phantom, fov_deg)
        export_npz(npz_dict)

    if visible_mode:
        fov_deg = 26.0
        visible_imgs, visible_poses, visible_angles = generate_projections(
            "./data/objs/test_ct/test_ct.obj",
            num_views=num_scans,
            image_size=(size, size),
            fov_deg=fov_deg,
        )
        npz_dict = get_npz_dict(
            visible_imgs,
            visible_poses,
            visible_angles,
            phantom,
            fov_deg=fov_deg,
        )
        export_npz(npz_dict)

def test_plot(ct_imgs, ct_poses):
    test_idx = np.random.randint(0, ct_imgs.shape[0])
    plt.title(f"Sinogram slice {test_idx}")
    plt.imshow(ct_imgs[test_idx])
    plt.colorbar()
    plt.show()

    # Rotate the phantom by the corresponding camera pose
    rotated_phantom = utils.rotate_phantom_by_pose(phantom, ct_poses[test_idx])

    # Visualize a central slice of the rotated phantom
    plt.figure()
    plt.title(f"Rotated Phantom Slice for Camera Pose {test_idx}")
    plt.imshow(rotated_phantom.sum(axis=1))
    plt.colorbar()
    plt.show()


# %%
# From ct nerf interp


# %%
def old3():
    # TODO : Generate the training set here with the identical features 
    #   Then DO NOT touch anything, train the model and import it into here, even 

    torch.cuda.empty_cache()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # model_path = "./logs/3-ct_data_13_320_64_741"
    # model, render_kwargs_test = get_model(model_path)

    phantom_idx = 13
    size = 256
    num_scans = 64
    assert num_scans % 2 == 0, "Number of scans must be even"
    hwf = (size, size, None)
    use_rgb = True
    
    phantom = ct_scan.load_phantom(phantom_idx, size)
    # phantom = np.ones_like(phantom) 
    euler_angles = np.array(
        [
            [0.0, theta, 0.0]
            for theta in np.linspace(0, np.pi, num_scans, endpoint=False)
        ]
    )
    scan_n = ct_scan.AstraScanVec3D(phantom.shape, euler_angles[::2], img_res=256)
    scan_2n = ct_scan.AstraScanVec3D(phantom.shape, euler_angles, img_res=256)
    ct_imgs = scan_2n.generate_ct_imgs(phantom)

    # NOTE : TODO : This is broken it seems to go the wrong way around 
    euler_poses = scan_2n.get_ct_camera_poses(radius=2.0) 
    # NOTE : DEBUG : QUICK FIX  
    euler_poses = euler_poses[::-1].copy()

    path_ds = create_nerf_ds(phantom_idx, phantom, ct_imgs, euler_poses, euler_angles)
    path_cfg = create_config(path_ds.name)
    # TODO : DEBUG : Right now this produces pure black images, 75 % guarntee its to width poses, 
    #   Fix by comparing the old way of getting and saving ct_poses, compare to current (i.e. should produce the same json) and then pick the working one 
    plt.imshow(ct_imgs[32])
    plt.show()
    # train_nerf(str(path_cfg))


def old():
    # Every 2nd image
    ct_imgs_even = ct_imgs[::2]
    ct_imgs_lanczos = ct_scan.lanczos_ct_imgs(ct_imgs_even)
    ct_imgs_lerp = ct_scan.lerp_ct_imgs(ct_imgs_even)
    ct_imgs_nerf = nerf_ct_imgs(ct_imgs_even, euler_poses, hwf, render_kwargs_test)
    ct_imgs_nerf_full = nerf_ct_imgs_full(
        ct_imgs_even, euler_poses, hwf, render_kwargs_test
    )

    ct_scan.plot_reconstructions(
        scan_2n,
        ct_imgs,
        phantom,
        size,
        title=f"[Full Orignial] Reconstruction from {num_scans} views",
    )
    ct_scan.plot_reconstructions(
        scan_n,
        ct_imgs_even,
        phantom,
        size,
        title=f"[Half Orignial] Reconstruction from {num_scans // 2} views",
    )
    ct_scan.plot_reconstructions(
        scan_2n,
        ct_imgs_lerp,
        phantom,
        size,
        title=f"[Half Orignial, Half Lerp] Reconstruction from {num_scans} views",
    )
    ct_scan.plot_reconstructions(
        scan_2n,
        ct_imgs_lanczos,
        phantom,
        size,
        title=f"[Half Orignial, Half lanczos] Reconstruction from {num_scans} views",
    )
    ct_scan.plot_reconstructions(
        scan_2n,
        ct_imgs_nerf,
        phantom,
        size,
        title=f"[Half Orignial, Half Nerf] Reconstruction from {num_scans} views",
    )
    ct_scan.plot_reconstructions(
        scan_2n,
        ct_imgs_nerf_full,
        phantom,
        size,
        title=f"[Full Nerf] Reconstruction from {num_scans} views",
    )

    # Render gifs
    RENDER_GIF = False
    if RENDER_GIF:
        utils.create_gif(
            images=ct_imgs_nerf,
            labels=np.round(euler_angles, 3),
            template_str = "Scan at angle {}",
            output_filename="./figs/temp/cfp1.gif",
            fps=8,
        )
        utils.create_gif(
            images=ct_imgs,
            labels=np.round(euler_angles, 3),
            template_str = "Scan at angle {}",
            output_filename="./figs/temp/cfp2.gif",
            fps=8,
        )

    visualize_camera_poses(scan_2n.get_ct_camera_poses())

# %%
def old():
    # Device setup.
    torch.cuda.empty_cache()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    model, render_kwargs_test = get_model("./logs/ct_data_13_320_64_741")

    # CT phantom and scan parameters.
    phantom_idx = 13
    size = 128
    num_scans = 16  # Control: use 16 scans.
    recon_iters = 48
    hwf = (size, size, None)

    min_theta_rad = 0.0
    max_theta_rad = 2 * np.pi

    phantom = ct_scan.load_phantom(phantom_idx, size)
    scan_params = ct_scan.AstraScanParameters(
        phantom_shape=phantom.shape,
        num_scans=num_scans,
        min_theta=min_theta_rad,
        max_theta=max_theta_rad,
    )

    # Generate original CT sinogram and normalized images.
    ct_imgs = scan_params.generate_ct_imgs(phantom)
    ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())
    angles_rad = scan_params.get_angles_rad()

    # Baseline CT reconstruction from original sinogram.
    sinogram = ct_imgs.swapaxes(0, 1)
    ct_recon = scan_params.reconstruct_3d_volume_alg(sinogram, recon_iters)

    # NeRF interpolation: generate intermediate images.
    ct_imgs_interp, angles_interp = interpolate_with_nerf(
        ct_imgs, min_theta_rad, max_theta_rad, hwf, render_kwargs_test
    )
    sinogram_interp = ct_imgs_interp.swapaxes(0, 1)
    # Prepare scan for interpolated sinogram.
    scan_params.num_scans = len(angles_interp)
    scan_params.re_init()
    ct_recon_interp = scan_params.reconstruct_3d_volume_alg(
        sinogram_interp, recon_iters
    )

    # Visualization: compare a representative CT slice and NeRF novel view.
    idx_slice = (size - 1) // 2
    mid_idx = (len(angles_rad) - 1) // 2
    novel_angle = utils.lerp(angles_rad[mid_idx], angles_rad[mid_idx + 1], 0.5)
    novel_view = utils.rgb_to_mono(
        get_image_at_theta(novel_angle, hwf, render_kwargs_test)
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    axes[0].imshow(ct_imgs[mid_idx])
    axes[0].set_title(f"CT View at {angles_rad[mid_idx]:.2f} Rad")
    axes[1].imshow(novel_view)
    axes[1].set_title(f"NeRF View at {novel_angle:.2f} Rad")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes[0].imshow(phantom[idx_slice])
    axes[0].set_title("GT Phantom Slice")
    axes[1].imshow(ct_recon[idx_slice])
    axes[1].set_title("SIRT Reconstruction (Original)")
    axes[2].imshow(ct_recon_interp[idx_slice])
    axes[2].set_title("SIRT Reconstruction (NeRF Interpolated)")
    plt.tight_layout()
    plt.show()

    # Compute PSNR for quantitative comparison.
    psnr_value = utils.psnr(ct_recon_interp, phantom)
    print(f"PSNR (Reconstructed vs. Phantom): {psnr_value:.2f} dB")

    # Optionally, create GIFs of the scans and reconstructions.
    render_gifs = False
    if render_gifs:
        print("Creating GIFs...")
        suffix = f"{phantom_idx}_{size}"
        prefix = "only_"
        utils.create_gif(
            ct_imgs_interp,
            np.round(angles_interp, 3),
            "Scan at angle {}",
            f"./figs/temp/{prefix}ct_nerf_interp_{suffix}.gif",
            fps=16,
        )
        utils.create_gif(
            ct_imgs,
            np.round(angles_rad, 3),
            "Scan at angle {}",
            f"./figs/temp/{prefix}ct_gt_{suffix}.gif",
            fps=8,
        )
        utils.create_gif(
            ct_recon_interp,
            np.arange(ct_recon_interp.shape[0]),
            "Reconstruction at slice {}",
            f"./figs/temp/{prefix}ct_recon_interp_{suffix}.gif",
            fps=12,
        )
        utils.create_gif(
            ct_recon,
            np.arange(ct_recon.shape[0]),
            "Reconstruction at slice {}",
            f"./figs/temp/{prefix}ct_recon_gt_{suffix}.gif",
            fps=12,
        )

    pass


# %%
