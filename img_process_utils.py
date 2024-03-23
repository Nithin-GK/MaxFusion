import controlnet_hinter

hint_func={
    'pose': controlnet_hinter.hint_openpose,
    'scribble': controlnet_hinter.hint_scribble,
    'canny': controlnet_hinter.hint_canny,
    'hed': controlnet_hinter.hint_hed,
    'depth': controlnet_hinter.hint_depth
}



def find_mode(hint, image):
    hint_fn = hint_func[hint]
    mode= hint_fn(image)
    return mode

