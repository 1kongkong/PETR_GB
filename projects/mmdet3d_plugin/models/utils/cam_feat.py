import torch

from ..._base import BaseModule


def feat_xy_grid(feat, corner_mode=False):
    """Generate the xy-grid of the feature map.

    Coordinate frame:
    -------> x
    |
    |
    y

    Args:
        feat (Tensor): the feature map whose shape should be [B, (N), C, H, W].
        corner_mode (bool): whether to use the corner mode. Default to False.
            If True, the shape of the output is [H+1, W+1, 2].

    Returns:
        Tensor: The x-y coordinates. The size of the tensor is [H, W, 2].
            If `corner_mode`, the shape of the output is [H+1, W+1, 2].
    """
    assert isinstance(feat, torch.Tensor)
    assert feat.ndim in (4, 5)
    H, W = list(feat.shape)[-2:]
    if corner_mode:
        H += 1
        W += 1
    grid_x = torch.linspace(0, W - 1, W, device=feat.device)
    grid_y = torch.linspace(0, H - 1, H, device=feat.device)
    grid_x = grid_x.view(1, W).repeat(H, 1)
    grid_y = grid_y.view(H, 1).repeat(1, W)
    grid = torch.stack((grid_x, grid_y), dim=2)
    return grid


def img_xy_grid(
    feat, img_h=None, img_w=None, shift_half_stride=False, corner_mode=False
):
    """Generate the xy-grid of the feature map. All values are pixel
    coordinates of the image. The feature map may be smaller than the image due
    to the downsampling of the image feature encoder.

    Coordinate frame:
    -------> x
    |
    |
    y

    Args:
        feat (Tensor): the feature map of the image. Its shape should be
            [B, (N), C, H, W].
        img_h (int): the height of the image. Default to None, and the height
            of the feature is used.
        img_w (int): the width of the image. Default to None, and the width
            of the feature is used.
        shift_half_stride (bool): left aligned or center-aligned.
            Default to False.
        corner_mode (bool): whether to use the corner mode. Default to False.
            If `corner_mode`, `shift_half_stride` must be False.

    Returns:
        Tensor: The x-y coordinates. The size of the tensor is [H, W, 2].
            If `corner_mode`, the shape of the output is [H+1, W+1, 2].
    """
    if corner_mode:
        assert not shift_half_stride

    grid = feat_xy_grid(feat, corner_mode=corner_mode)
    H, W = list(feat.shape)[-2:]
    if img_h is None:
        img_h = H
    if img_w is None:
        img_w = W
    h_stride = img_h / H
    w_stride = img_w / W

    if img_w != W:
        grid[:, :, 0] = grid[:, :, 0] * w_stride
    if img_h != H:
        grid[:, :, 1] = grid[:, :, 1] * h_stride

    if shift_half_stride:
        grid[:, :, 0] = grid[:, :, 0] + w_stride / 2
        grid[:, :, 1] = grid[:, :, 1] + h_stride / 2

    return grid


def cam_xyz_grid(
    cam_feat,
    cam_intrinsics,
    cam_intrinsics_adj=None,
    img_h=None,
    img_w=None,
    shift_half_stride=False,
    corner_mode=False,
    camera_type="pinhole",
):
    """Generate the xyz-grid of the feature map. All values are in the camera
    coordinate frame.

    Pixel coordinate frame:
    -------> x
    |
    |
    y

    Camera coordinate frame:
        z (forward)
       /
      /
     /
    ---------> x (rigntward)
    |
    |
    y (downward)

    Args:
        cam_feat (Tensor): the feature map of the image. Its shape should be
            [B, (N), C, H, W]. It can be a list/tuple.
        cam_intrinsics (Tensor): The intrinsics of the cameras.
            The cam_intrinsics should be a 4D Tensor or 3D Tensor.
            The shape should be [B, N, 3, 3] or [B, 3, 3].
            D [u_ori, v_ori, 1] = K @ [X_cam, Y_cam, Z_cam]
        cam_intrinsics_adj (Tensor, optional): Adjustion of the camera
            intrinsics deriving from the data augmentation/preprocessing.
            The cam_intrinsics_adj should be a 4D Tensor or 3D Tensor.
            The shape should be [B, N, 3, 3] or [B, 3, 3].
            Defaults to None.
            [u_input, v_input, 1] = A @ [u_ori, v_ori, 1]
        img_h (int): the height of the image. Default to None, and the height
            of the feature is used.
        img_w (int): the width of the image. Default to None, and the width
            of the feature is used.
        shift_half_stride (bool): left aligned or center-aligned.
            Default to False.
        corner_mode (bool): whether to use the corner mode. Default to False.
        camera_type (str, optional): The type of the cameras.
            Currently `pinhole` and `cylinder` are supported.
            Defaults to `pinhole`.

    Returns:
        Tensor: The x-y-z coordinates in camera coordinate system.
            The size of the tensor is [B, (N,) H, W, 3].
            If `corner_mode`, the shape of the output is [B, (N,) H+1, W+1, 3].
    """
    if isinstance(cam_feat, (list, tuple)):
        return [
            cam_xyz_grid(
                each,
                cam_intrinsics,
                cam_intrinsics_adj=cam_intrinsics_adj,
                img_h=img_h,
                img_w=img_w,
                shift_half_stride=shift_half_stride,
                corner_mode=corner_mode,
                camera_type=camera_type,
            )
            for each in cam_feat
        ]

    assert camera_type in ("pinhole", "cylinder")

    uv_input = img_xy_grid(
        cam_feat,
        img_h=img_h,
        img_w=img_w,
        shift_half_stride=shift_half_stride,
        corner_mode=corner_mode,
    )

    if cam_feat.ndim == 4:
        B = cam_feat.size(0)
        N = 0
        assert isinstance(cam_intrinsics, torch.Tensor)
        assert cam_intrinsics.size() == torch.Size([B, 3, 3])
        if cam_intrinsics_adj is not None:
            assert isinstance(cam_intrinsics_adj, torch.Tensor)
            assert cam_intrinsics_adj.size() == torch.Size([B, 3, 3])
    elif cam_feat.ndim == 5:
        B = cam_feat.size(0)
        N = cam_feat.size(1)
        cam_feat = cam_feat.view(B * N, *list(cam_feat.shape[2:]))
        assert isinstance(cam_intrinsics, torch.Tensor)
        assert cam_intrinsics.size() == torch.Size([B, N, 3, 3])
        cam_intrinsics = cam_intrinsics.view(B * N, 3, 3)
        if cam_intrinsics_adj is not None:
            assert isinstance(cam_intrinsics_adj, torch.Tensor)
            assert cam_intrinsics_adj.size() == torch.Size([B, N, 3, 3])
            cam_intrinsics_adj = cam_intrinsics_adj.view(B * N, 3, 3)
    S = cam_feat.size(0)
    H_grid = uv_input.size(0)
    W_grid = uv_input.size(1)

    uv1_input = torch.cat((uv_input, uv_input.new_ones((H_grid, W_grid, 1))), dim=-1)
    uv1_input = uv1_input.unsqueeze(0).expand(S, -1, -1, -1)  # [S, H, W, 2]
    uv1_input = uv1_input.view(S, H_grid * W_grid, 3)

    if cam_intrinsics_adj is None:
        uv1_cam = uv1_input
    else:
        # [u_input, v_input, 1] = A @ [u_ori, v_ori, 1]
        # [u_ori, v_ori, 1] = inv(A) @ [u_input, v_input, 1]
        # [u_ori, v_ori, 1].T = [u_input, v_input, 1].T @ inv(A).T
        # # NOTE: It is always preferred to use solve(), not inv().
        # # https://pytorch.org/docs/stable/generated/torch.linalg.solve.html
        # uv1_cam = torch.linalg.solve(
        #     cam_intrinsics_adj,
        #     uv1_input.permute(0, 2, 1),
        # ).permute(0, 2, 1)
        uv1_cam = uv1_input @ torch.linalg.inv(cam_intrinsics_adj).permute(0, 2, 1)

    if camera_type == "pinhole":
        # D[u_ori, v_ori, 1] = K @ [X_cam, Y_cam, Z_cam]
        # [X_cam, Y_cam, 1] = inv(K) @ [u_ori, v_ori, 1]
        # [X_cam, Y_cam, 1].T = [u_ori, v_ori, 1].T @ inv(K).T
        # # NOTE: It is always preferred to use solve(), not inv().
        # # https://pytorch.org/docs/stable/generated/torch.linalg.solve.html
        # xyz_cam = torch.linalg.solve(
        #     cam_intrinsics,
        #     uv1_cam.permute(0, 2, 1),
        # ).permute(0, 2, 1)
        xyz_cam = uv1_cam @ torch.linalg.inv(cam_intrinsics).permute(0, 2, 1)
    elif camera_type == "cylinder":
        # D[u_ori, v_ori, 1] = K @ [theta_cam, Y_cam, R_cam]
        # [theta_cam, Y_cam, R_cam] = inv(K) @ [u_ori, v_ori, 1]
        # [theta_cam, Y_cam, R_cam].T = [u_ori, v_ori, 1].T @ inv(K).T
        # # NOTE: It is always preferred to use solve(), not inv().
        # # https://pytorch.org/docs/stable/generated/torch.linalg.solve.html
        # theta_y_1_tmp = torch.linalg.solve(
        #     cam_intrinsics,
        #     uv1_cam.permute(0, 2, 1),
        # ).permute(0, 2, 1)
        theta_y_1_tmp = uv1_cam @ torch.linalg.inv(cam_intrinsics).permute(0, 2, 1)
        xyz_cam = torch.cat(
            (
                torch.sin(theta_y_1_tmp[..., :1]),
                theta_y_1_tmp[..., 1:2],
                torch.cos(theta_y_1_tmp[..., :1]),
            ),
            dim=-1,
        )
    else:
        raise NotImplementedError(
            "Currently only `pinhole` and `cylinder`" " camera are supported."
        )
    if N > 0:
        xyz_cam = xyz_cam.view(B, N, H_grid, W_grid, 3)
    else:
        xyz_cam = xyz_cam.view(B, H_grid, W_grid, 3)
    return xyz_cam


def ego_trans_grid_of_cam_feat_obs(
    cam_feat,
    cam_extrinsics,
    cam_intrinsics,
    cam_intrinsics_adj=None,
    img_h=None,
    img_w=None,
    shift_half_stride=False,
    corner_mode=False,
    camera_type="pinhole",
    apply_fl_to_obs=False,
):
    """Generate the xy-grid of the feature map. All values are pixel
    coordinates of the image. The feature map may be smaller than the image due
    to the downsampling of the image feature encoder.

    Pixel coordinate frame:
    -------> x
    |
    |
    y

    Camera coordinate frame:
        z (forward)
       /
      /
     /
    ---------> x (rigntward)
    |
    |
    y (downward)

    Ego-vehicle coordinate frame:
    (upward) z    x (forward)
             ^   /
             |  /
             | /
    y<-------O
    (leftward)

    Args:
        cam_feat (Tensor): the feature map of the image. Its shape should be
            [B, (N), C, H, W]. It can be a list/tuple.
        cam_extrinsics (Tensor): The extrinsics of the cameras.
            The cam_extrinsics should be a 4D Tensor or 3D Tensor.
            The shape should be [B, N, 4, 4] or [B, 4, 4].
            [X_cam, Y_cam, Z_cam, 1] = P @ [X_ego, Y_ego, Z_ego, 1]
            NOTE: `cam_extrinsics` should be checked carefully.
        cam_intrinsics (Tensor): The intrinsics of the cameras.
            The cam_intrinsics should be a 4D Tensor or 3D Tensor.
            The shape should be [B, N, 3, 3] or [B, 3, 3].
            D [u_ori, v_ori, 1] = K @ [X_cam, Y_cam, Z_cam]
        cam_intrinsics_adj (Tensor, optional): Adjustion of the camera
            intrinsics deriving from the data augmentation/preprocessing.
            The cam_intrinsics_adj should be a 4D Tensor or 3D Tensor.
            The shape should be [B, N, 3, 3] or [B, 3, 3].
            Defaults to None.
            [u_input, v_input, 1] = A @ [u_ori, v_ori, 1]
        img_h (int): the height of the image. Default to None, and the height
            of the feature is used.
        img_w (int): the width of the image. Default to None, and the width
            of the feature is used.
        shift_half_stride (bool): left aligned or center-aligned.
            Default to False.
        corner_mode (bool): whether to use the corner mode. Default to False.
        camera_type (str, optional): The type of the cameras.
            Currently `pinhole` and `cylinder` are supported.
            Defaults to `pinhole`.

    Returns:
        Tensor: The coordinates in ego-vehicle coordinate system.
            The size of the tensor is [B, (N,) H, W, 6].
            The 6 values are (x, y, z, rot_x, rot_y, rot_z),
            where (x, y, z) is the location of the camera optical center.
            (rot_x, rot_y, rot_z) is the spherical coordinates of the ray.
            |(rot_x, rot_y, rot_z)| == 1.
            If `corner_mode`, The size of the tensor is [B, (N,) H, W, 15].
            The 15 values are (x, y, z, rot_A, rot_B, rot_C, rot_D].
            The rot_A, rot_B, rot_C, rot_D are for four corner points:
            A --- B
            |     |
            C --- D
    """
    if isinstance(cam_feat, (list, tuple)):
        return [
            ego_trans_grid_of_cam_feat_obs(
                each,
                cam_extrinsics,
                cam_intrinsics,
                cam_intrinsics_adj=cam_intrinsics_adj,
                img_h=img_h,
                img_w=img_w,
                shift_half_stride=shift_half_stride,
                corner_mode=corner_mode,
                camera_type=camera_type,
                apply_fl_to_obs=apply_fl_to_obs,
            )
            for each in cam_feat
        ]
    # [B, (N), H, W, 3]
    xyz_cam = cam_xyz_grid(
        cam_feat,
        cam_intrinsics,
        cam_intrinsics_adj=cam_intrinsics_adj,
        img_h=img_h,
        img_w=img_w,
        shift_half_stride=shift_half_stride,
        corner_mode=corner_mode,
        camera_type=camera_type,
    )
    H = xyz_cam.size(-3)
    W = xyz_cam.size(-2)
    if xyz_cam.ndim == 4:
        B = xyz_cam.size(0)
        N = 0
        xyz_cam = xyz_cam.view(B, H * W, 3)
        assert isinstance(cam_extrinsics, torch.Tensor)
        assert cam_extrinsics.size() == torch.Size([B, 4, 4])
    elif xyz_cam.ndim == 5:
        B = xyz_cam.size(0)
        N = xyz_cam.size(1)
        xyz_cam = xyz_cam.view(B * N, H * W, 3)
        assert isinstance(cam_extrinsics, torch.Tensor)
        assert cam_extrinsics.size() == torch.Size([B, N, 4, 4])
        cam_extrinsics = cam_extrinsics.view(B * N, 4, 4)
    S = xyz_cam.size(0)

    xyz1_cam = torch.cat([xyz_cam, xyz_cam.new_ones((S, H * W, 1))], dim=-1)
    # [X_cam, Y_cam, Z_cam, 1] = P @ [X_ego, Y_ego, Z_ego, 1]
    # [X_ego, Y_ego, Z_ego, 1] = inv(P) @ [X_cam, Y_cam, Z_cam, 1]
    # [X_ego, Y_ego, Z_ego, 1].T = [X_cam, Y_cam, Z_cam, 1].T @ inv(P).T
    # # NOTE: It is always preferred to use solve(), not inv().
    # # https://pytorch.org/docs/stable/generated/torch.linalg.solve.html
    # xyz1_ego = torch.linalg.solve(
    #     cam_extrinsics,
    #     xyz1_cam.permute(0, 2, 1),
    # ).permute(0, 2, 1)
    # ori1_cam = torch.zeros((S, 4),
    #                        dtype=xyz1_cam.dtype,
    #                        device=xyz1_cam.device)
    # ori1_cam[:, -1] = 1
    # ori1_ego = torch.linalg.solve(cam_extrinsics, ori1_cam)  # [S, 4]
    ext_inv_trans = torch.linalg.inv(cam_extrinsics).permute(0, 2, 1)
    xyz1_ego = xyz1_cam @ ext_inv_trans
    ori1_ego = ext_inv_trans[:, 3, :]  # [S, 4]

    if N > 0:
        xyz_ego = xyz1_ego.view(B, N, H, W, 4)[:, :, :, :, :3].contiguous()
        ori_ego = ori1_ego.view(B, N, 1, 1, 4)[:, :, :, :, :3].contiguous()
        ori_ego = ori_ego.expand(-1, -1, H, W, -1)
    else:
        xyz_ego = xyz1_ego.view(B, H, W, 4)[:, :, :, :3].contiguous()
        ori_ego = ori1_ego.view(B, 1, 1, 4)[:, :, :, :3].contiguous()
        ori_ego = ori_ego.expand(-1, H, W, -1)
    obs_rot = xyz_ego - ori_ego
    obs_rot_norm = torch.linalg.vector_norm(obs_rot, dim=-1, keepdim=True)
    obs_rot = obs_rot / obs_rot_norm
    if apply_fl_to_obs:
        fx = cam_intrinsics[..., 0, 0]
        fx = fx / fx.max(-1)[0].unsqueeze(-1)  # normalize
        obs_rot = obs_rot * fx.view(B, N, 1, 1, 1)
    obs_shift = ori_ego
    if corner_mode:
        obs_rot = [
            obs_rot[..., : H - 1, : W - 1, :],
            obs_rot[..., : H - 1, 1:W, :],
            obs_rot[..., 1:H, : W - 1, :],
            obs_rot[..., 1:H, 1:W, :],
        ]
        obs_rot = torch.cat(obs_rot, dim=-1)
        obs_shift = obs_shift[..., : H - 1, : W - 1, :]
    obs_ego_trans = torch.cat([obs_shift, obs_rot], dim=-1)
    return obs_ego_trans


def cam_intrinsic_code(cam_intrinsics, cam_intrinsics_adj=None, camera_type="pinhole"):
    """Generate the raw code for camera instrisics. `fx` and `fy` are used.

    Args:
        cam_intrinsics (Tensor): The intrinsics of the cameras.
            The cam_intrinsics should be a 4D Tensor or 3D Tensor.
            The shape should be [B, N, 3, 3] or [B, 3, 3].
            D [u_ori, v_ori, 1] = K @ [X_cam, Y_cam, Z_cam]
        cam_intrinsics_adj (Tensor, optional): Adjustion of the camera
            intrinsics deriving from the data augmentation/preprocessing.
            The cam_intrinsics_adj should be a 4D Tensor or 3D Tensor.
            The shape should be [B, N, 3, 3] or [B, 3, 3].
            Defaults to None.
            [u_input, v_input, 1] = A @ [u_ori, v_ori, 1]
        camera_type (str, optional): The type of the cameras.
            Currently `pinhole` and `cylinder` are supported.
            Defaults to `pinhole`.

    Returns:
        Tensor: The x-y-1 coordinates in camera coordinate system.
            The size of the tensor is [B, (N,) 2]. The two values
            are (fx, fy).
    """
    assert camera_type in ("pinhole", "cylinder")

    assert isinstance(cam_intrinsics, torch.Tensor)
    if cam_intrinsics.ndim == 3:
        B = cam_intrinsics.size(0)
        N = 0
        assert cam_intrinsics.size() == torch.Size([B, 3, 3])
        if cam_intrinsics_adj is not None:
            assert isinstance(cam_intrinsics_adj, torch.Tensor)
            assert cam_intrinsics_adj.size() == torch.Size([B, 3, 3])
    elif cam_intrinsics.ndim == 4:
        B = cam_intrinsics.size(0)
        N = cam_intrinsics.size(1)
        assert cam_intrinsics.size() == torch.Size([B, N, 3, 3])
        cam_intrinsics = cam_intrinsics.view(B * N, 3, 3)
        if cam_intrinsics_adj is not None:
            assert isinstance(cam_intrinsics_adj, torch.Tensor)
            assert cam_intrinsics_adj.size() == torch.Size([B, N, 3, 3])
            cam_intrinsics_adj = cam_intrinsics_adj.view(B * N, 3, 3)
    else:
        raise RuntimeError(
            "The shape of the `cam_intrinsics` should" " be [B, N, 3, 3] or [B, 3, 3]"
        )
    if cam_intrinsics_adj is not None:
        # D[u_ori, v_ori, 1] = K @ [X_cam, Y_cam, Z_cam]
        # [u_input, v_input, 1] = A @ [u_ori, v_ori, 1]
        # D[u_input, v_input, 1] = A @ K @ [X_cam, Y_cam, Z_cam]
        cam_intrinsics = cam_intrinsics_adj @ cam_intrinsics

    if N > 0:
        cam_intrinsics = cam_intrinsics.view(B, N, 3, 3)

    if camera_type in ["pinhole", "cylinder"]:
        fx = cam_intrinsics[..., 0, 0]
        fy = cam_intrinsics[..., 1, 1]
        code = torch.stack([fx, fy], dim=-1)  # [B, (N,) 2]
    else:
        raise NotImplementedError(
            "Currently only `pinhole` and `cylinder`" " camera are supported."
        )

    return code


class CamFeatPosEncoder(BaseModule):
    """Postional embedding generator for camere features."""

    def __init__(
        self,
        shift_half_stride=False,
        corner_mode=False,
        camera_type="pinhole",
        with_stride=False,
        concat_feat=True,
        init_cfg=None,
        freeze_weights=False,
        freeze_bn_stat=False,
    ):
        """Module initialization.

        Args:
            shift_half_stride (bool, optional): On index mapping.
                False: pixel_index = feature_index * stride.
                True:  pixel_index = feature_index * stride + stride / 2.
                Defaults to False.
            corner_mode (bool): whether to use the corner mode.
                Default to False.
            camera_type (str, optional): The type of the camera.
                Defaults to `pinhole`.
            with_stride (bool, optional): The stride_h and stride_w of the
                feature are used as positonal embedding. Defaults to False.
            concat_feat (bool, optional): Whether to concatanate the feature
                and the positional embedding in channel dimension.
                True: return `feature` || `positonal embedding`.
                False: return only `positonal embedding`.
                Defaults to True.
        """
        super().__init__(
            init_cfg=init_cfg,
            freeze_weights=freeze_weights,
            freeze_bn_stat=freeze_bn_stat,
        )
        self._shift_half_stride = shift_half_stride
        self._corner_mode = corner_mode
        self._camera_type = camera_type
        self._with_stride = with_stride
        self._concat_feat = concat_feat

    @staticmethod
    def concat_intrinsic_code(ori_code, intrinsic_code):
        """Concatanate the `ori_code` and `intrinsic_code`.

        Args:
            ori_code (Tensor): The shape should be [B, (N,) H, W, C1]
            intrinsic_code (Tensor): The shape should be [B, (N,) C2]
        """
        H = ori_code.size(-3)
        W = ori_code.size(-2)
        intrinsic_code = intrinsic_code.unsqueeze(-2).unsqueeze(-2)
        if intrinsic_code.ndim == 4:
            intrinsic_code = intrinsic_code.expand(-1, H, W, -1)
        else:
            intrinsic_code = intrinsic_code.expand(-1, -1, H, W, -1)
        code = torch.cat([ori_code, intrinsic_code], dim=-1)
        return code

    @staticmethod
    def concat_stride(ori_code, img_h, img_w):
        """Concatanate the `ori_code` with the stride_h and stride_w.

        Args:
            ori_code (Tensor): The shape should be [B, (N,) H, W, C1]
            img_h (int): The height of the input image.
            img_w (int): The width of the input image.
        """
        H = ori_code.size(-3)
        W = ori_code.size(-2)
        stride_h = img_h / H
        stride_w = img_w / W
        shape = list(ori_code.size()[:-1]) + [1]
        stride_h = ori_code.new_full(shape, stride_h)
        stride_w = ori_code.new_full(shape, stride_w)
        code = torch.cat([ori_code, stride_h, stride_w], dim=-1)
        return code

    @staticmethod
    def permute_code(code):
        """Permute the `code`. The original shape should be [B, (N,) H, W, C]
        and the output shape should be [B, (N,) C, H, W]

        Args:
            code (Tensor): The shape should be [B, (N,) H, W, C]
        """
        if code.ndim == 4:
            return code.permute(0, 3, 1, 2).contiguous()
        else:
            return code.permute(0, 1, 4, 2, 3).contiguous()

    def forward(
        self,
        cam_feat,
        cam_extrinsics,
        cam_intrinsics,
        cam_intrinsics_adj=None,
        img_h=None,
        img_w=None,
    ):
        code = ego_trans_grid_of_cam_feat_obs(
            cam_feat,
            cam_extrinsics,
            cam_intrinsics,
            cam_intrinsics_adj=cam_intrinsics_adj,
            img_h=img_h,
            img_w=img_w,
            shift_half_stride=self._shift_half_stride,
            corner_mode=self._corner_mode,
            camera_type=self._camera_type,
        )

        intr = cam_intrinsic_code(
            cam_intrinsics,
            cam_intrinsics_adj=cam_intrinsics_adj,
            camera_type=self._camera_type,
        )
        if isinstance(code, (list, tuple)):
            code = [self.concat_intrinsic_code(c, intr) for c in code]
        else:
            code = self.concat_intrinsic_code(code, intr)

        if self._with_stride:
            assert img_h is not None
            assert img_w is not None
            if isinstance(code, (list, tuple)):
                code = [self.concat_stride(c, img_h, img_w) for c in code]
            else:
                code = self.concat_stride(code, img_h, img_w)

        if isinstance(code, (list, tuple)):
            code = [self.permute_code(c) for c in code]
        else:
            code = self.permute_code(code)

        if self._concat_feat:
            if isinstance(code, (list, tuple)):
                return [torch.cat([f, c], dim=-3) for f, c in zip(cam_feat, code)]
            else:
                return torch.cat([cam_feat, code], dim=-3)
        else:
            return code


class CamFeatPosEncoderV2(BaseModule):
    """Postional embedding generator for camere features."""

    def __init__(
        self,
        shift_half_stride=False,
        corner_mode=False,
        camera_type="pinhole",
        with_intrinsic=True,
        with_stride=True,
        apply_fl_to_obs=False,
    ):
        """Module initialization.

        Args:
            shift_half_stride (bool, optional): On index mapping.
                False: pixel_index = feature_index * stride.
                True:  pixel_index = feature_index * stride + stride / 2.
                Defaults to False.
            corner_mode (bool): whether to use the corner mode.
                Default to False.
            camera_type (str, optional): The type of the camera.
                Defaults to `pinhole`.
            with_intrinsic (bool, optional): Whether to return the PE
                of camera intrinsic fx/fy. Defaults to True.
            with_stride (bool, optional): Whether to return the PE
                of feature stride_h and stride_w. Defaults to True.
        """
        super().__init__()
        self._shift_half_stride = shift_half_stride
        self._corner_mode = corner_mode
        self._camera_type = camera_type
        self._with_intrinsic = with_intrinsic
        self._with_stride = with_stride
        self._apply_fl_to_obs = apply_fl_to_obs

    @staticmethod
    def permute_code(code):
        """Permute the `code`.

        The original shape should be [B, (N,) H, W, C] and the output shape
        should be [B, (N,) C, H, W].
        """
        if code.ndim == 4:
            return code.permute(0, 3, 1, 2).contiguous()
        else:
            return code.permute(0, 1, 4, 2, 3).contiguous()

    @staticmethod
    def repeat_intrinsic_code(feat, intrinsic_code):
        H = feat.size(-2)
        W = feat.size(-1)
        intrinsic_code = intrinsic_code.unsqueeze(-1).unsqueeze(-1)
        if intrinsic_code.ndim == 4:
            intrinsic_code = intrinsic_code.expand(-1, -1, H, W)
        else:
            intrinsic_code = intrinsic_code.expand(-1, -1, -1, H, W)
        return intrinsic_code

    @staticmethod
    def encode_stride(feat, img_h, img_w):
        H = feat.size(-2)
        W = feat.size(-1)
        stride_h = img_h / H
        stride_w = img_w / W
        stride_h = feat.new_full(feat[..., 0:1, :, :].size(), stride_h)
        stride_w = feat.new_full(feat[..., 0:1, :, :].size(), stride_w)
        return torch.cat([stride_h, stride_w], dim=-3)

    def forward(
        self,
        cam_feat,
        cam_extrinsics,
        cam_intrinsics,
        cam_intrinsics_adj=None,
        img_h=None,
        img_w=None,
    ):
        """Generate the positional embedding.

        Returns:
            dict: The positional embedding.
            {
                'obs'(list[Tensor]|Tensor): shape [B, (N,) 6, H, W],
                'intrinsic'(list[Tensor]|Tensor): shape [B, (N,) 2, H, W],
                'stride'(list[Tensor]|Tensor): shape [B, (N,) 2, H, W],
            }
        """

        codes = {}

        obs = ego_trans_grid_of_cam_feat_obs(
            cam_feat,
            cam_extrinsics,
            cam_intrinsics,
            cam_intrinsics_adj=cam_intrinsics_adj,
            img_h=img_h,
            img_w=img_w,
            shift_half_stride=self._shift_half_stride,
            corner_mode=self._corner_mode,
            camera_type=self._camera_type,
            apply_fl_to_obs=self._apply_fl_to_obs,
        )
        if isinstance(obs, (list, tuple)):
            obs = [self.permute_code(o) for o in obs]
        else:
            obs = self.permute_code(obs)
        codes["obs"] = obs

        if self._with_intrinsic:
            intr = cam_intrinsic_code(
                cam_intrinsics,
                cam_intrinsics_adj=cam_intrinsics_adj,
                camera_type=self._camera_type,
            )
            if isinstance(cam_feat, (list, tuple)):
                intr = [self.repeat_intrinsic_code(f, intr) for f in cam_feat]
            else:
                intr = self.repeat_intrinsic_code(cam_feat, intr)
            codes["intrinsic"] = intr

        if self._with_stride:
            assert img_h is not None
            assert img_w is not None
            if isinstance(cam_feat, (list, tuple)):
                stride = [self.encode_stride(f, img_h, img_w) for f in cam_feat]
            else:
                stride = self.encode_stride(cam_feat, img_h, img_w)
            codes["stride"] = stride

        return codes
