import stir


def convert_3D_ProjData_to_2D_from_files(output_filename, ProjData_3D_filename, ProjData_2D_template,
                                         axial_pos_num=0, segment_num=0):
    """
    Convert a 3D projection data to 2D projection data. The 2D projection data is saved in a file.
    :param output_filename: output filename of the 2D projection data
    :param ProjData_3D_filename: filename of the 3D projection data
    :param ProjData_2D_template: filename of the 2D projection data template
    :param axial_pos_num: axial position number of the 3D projection data to extract
    :param segment_num: segment number of the 3D projection data to extract
    :return: None
    """

    # Load data
    projdata3D = stir.ProjData_read_from_file(ProjData_3D_filename)
    projdata2D = stir.ProjData_read_from_file(ProjData_2D_template)

    # extract 2D sinogram from 3D projection data
    projdata2DInMem = extract_sinogram_from_3D_into_2D(projdata3D, projdata2D, axial_pos_num, segment_num)

    # Save data to file
    projdata2DInMem.write_to_file(output_filename)


def extract_sinogram_from_3D_into_2D(projdata3D, projdata2D, axial_pos_num, segment_num):
    """
    Convert a 3D projection data to 2D projection data in memory. The 2D projection data in memory is returned.
    :param projdata3D: stir.ProjData object of 3D projection data
    :param projdata2D: stir.ProjData object of 2D projection data template
    :param axial_pos_num: axial position number of the 3D projection data to extract
    :param segment_num: segment number of the 3D projection data to extract
    :return: 2D projection data in memory of given axial position and segment number
    """
    # Get sinogram from 3D projection data
    sinogram_to_copy = projdata3D.get_sinogram(axial_pos_num, segment_num)
    # Get 2D projection data template in memory to perform the conversion
    projdata2DInMem = stir.ProjDataInMemory(projdata2D.get_exam_info(), projdata2D.get_proj_data_info())
    # Get 2D projection data sinogram (the only one) and fill with sino
    target_sino = projdata2DInMem.get_sinogram(0, 0)
    target_sino.fill(sinogram_to_copy.flat())
    # set target sino to back into projdata2DInMem with new values
    projdata2DInMem.set_sinogram(target_sino)
    return projdata2DInMem


def main():

    axial_pos_num = 12
    segment_num = 0
    data_configs = {
        "prompts":
            {
                "output_filename": "my_Prompts2d.hs",
                "ProjData_3D_filename": "../my_Prompts.hs",
                "ProjData_2D_template": "template2D.hs",
                "axial_pos_num": axial_pos_num,
                "segment_num": segment_num
            },
        "additive":
            {
                "output_filename": "my_total_additive_52d.hs",
                "ProjData_3D_filename": "../my_total_additive_5.hs",
                "ProjData_2D_template": "template2D.hs",
                "axial_pos_num": axial_pos_num,
                "segment_num": segment_num
            },
        "mutlfactors":
            {
                "output_filename": "my_multfactors2d.hs",
                "ProjData_3D_filename": "../my_multfactors.hs",
                "ProjData_2D_template": "template2D.hs",
                "axial_pos_num": axial_pos_num,
                "segment_num": segment_num
            },
    }
    for item in data_configs:
        convert_3D_ProjData_to_2D_from_files(**data_configs[item])


if __name__ == "__main__":
    main()
