function(FindStyleHeaders path style_class file_pattern headers)
    file(GLOB files "${path}/${file_pattern}*.h")
    get_property(hlist GLOBAL PROPERTY ${headers})

    foreach(file_name ${files})
        file(STRINGS ${file_name} is_style LIMIT_COUNT 1 REGEX ${style_class})
        if(is_style)
            list(APPEND hlist ${file_name})
        endif()
    endforeach()
    set_property(GLOBAL PROPERTY ${headers} "${hlist}")
endfunction(FindStyleHeaders)

function(AddStyleHeader path headers)
    get_property(hlist GLOBAL PROPERTY ${headers})
    list(APPEND hlist ${path})
    set_property(GLOBAL PROPERTY ${headers} "${hlist}")
endfunction(AddStyleHeader)

function(FindStyleHeadersExt path style_class extension headers sources)
    get_property(hlist GLOBAL PROPERTY ${headers})
    get_property(slist GLOBAL PROPERTY ${sources})
    set(ext_list)
    get_filename_component(abs_path "${path}" ABSOLUTE)

    foreach(file_name ${hlist})
        get_filename_component(basename ${file_name} NAME_WE)
        set(ext_file_name "${abs_path}/${basename}_${extension}.h")
        if(EXISTS "${ext_file_name}")
            file(STRINGS ${ext_file_name} is_style LIMIT_COUNT 1 REGEX ${style_class})
            if(is_style)
                list(APPEND ext_list ${ext_file_name})

                set(source_file_name "${abs_path}/${basename}_${extension}.cpp")
                if(EXISTS "${source_file_name}")
                    list(APPEND slist ${source_file_name})
                endif()
            endif()
        endif()
    endforeach()

    list(APPEND hlist ${ext_list})
    set_property(GLOBAL PROPERTY ${headers} "${hlist}")
    set_property(GLOBAL PROPERTY ${sources} "${slist}")
endfunction(FindStyleHeadersExt)

function(CreateStyleHeader path filename)
    math(EXPR N "${ARGC}-2")

    set(temp "")
    if(N GREATER 0)
        math(EXPR ARG_END   "${ARGC}-1")
 
        foreach(IDX RANGE 2 ${ARG_END})
            list(GET ARGV ${IDX} FNAME)
            get_filename_component(FNAME ${FNAME} NAME)
            set(temp "${temp}#include \"${FNAME}\"\n")
        endforeach()
    endif()
    message(STATUS "Generating ${filename}...")
    file(WRITE "${path}/${filename}.tmp" "${temp}" )
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${path}/${filename}.tmp" "${path}/${filename}")
endfunction(CreateStyleHeader)

function(GenerateStyleHeader path property style)
    get_property(files GLOBAL PROPERTY ${property})
    #message("${property} = ${files}")
    CreateStyleHeader("${path}" "style_${style}.h" ${files})
endfunction(GenerateStyleHeader)

function(RegisterNBinStyles search_path)
    FindStyleHeaders(${search_path} NBIN_CLASS      nbin_      NBIN      ) # nbin      ) # neighbor
endfunction(RegisterNBinStyles)

function(RegisterNPairStyles search_path)
    FindStyleHeaders(${search_path} NPAIR_CLASS     npair_     NPAIR     ) # npair     ) # neighbor
endfunction(RegisterNPairStyles)

function(RegisterNBinStyle path)
    AddStyleHeader(${path} NBIN)
endfunction(RegisterNBinStyle)

function(RegisterNPairStyle path)
    AddStyleHeader(${path} NPAIR)
endfunction(RegisterNPairStyle)

function(RegisterStyles search_path)
    FindStyleHeaders(${search_path} ANGLE_CLASS     angle_     ANGLE     ) # angle     ) # force
    FindStyleHeaders(${search_path} ATOM_CLASS      atom_vec_  ATOM_VEC  ) # atom      ) # atom      atom_vec_hybrid
    FindStyleHeaders(${search_path} BODY_CLASS      body_      BODY      ) # body      ) # atom_vec_body
    FindStyleHeaders(${search_path} BOND_CLASS      bond_      BOND      ) # bond      ) # force
    FindStyleHeaders(${search_path} COMMAND_CLASS   ""         COMMAND   ) # command   ) # input
    FindStyleHeaders(${search_path} COMPUTE_CLASS   compute_   COMPUTE   ) # compute   ) # modify
    FindStyleHeaders(${search_path} DIHEDRAL_CLASS  dihedral_  DIHEDRAL  ) # dihedral  ) # force
    FindStyleHeaders(${search_path} DUMP_CLASS      dump_      DUMP      ) # dump      ) # output    write_dump
    FindStyleHeaders(${search_path} FIX_CLASS       fix_       FIX       ) # fix       ) # modify
    FindStyleHeaders(${search_path} IMPROPER_CLASS  improper_  IMPROPER  ) # improper  ) # force
    FindStyleHeaders(${search_path} INTEGRATE_CLASS ""         INTEGRATE ) # integrate ) # update
    FindStyleHeaders(${search_path} KSPACE_CLASS    ""         KSPACE    ) # kspace    ) # force
    FindStyleHeaders(${search_path} MINIMIZE_CLASS  min_       MINIMIZE  ) # minimize  ) # update
    FindStyleHeaders(${search_path} NBIN_CLASS      nbin_      NBIN      ) # nbin      ) # neighbor
    FindStyleHeaders(${search_path} NPAIR_CLASS     npair_     NPAIR     ) # npair     ) # neighbor
    FindStyleHeaders(${search_path} NSTENCIL_CLASS  nstencil_  NSTENCIL  ) # nstencil  ) # neighbor
    FindStyleHeaders(${search_path} NTOPO_CLASS     ntopo_     NTOPO     ) # ntopo     ) # neighbor
    FindStyleHeaders(${search_path} PAIR_CLASS      pair_      PAIR      ) # pair      ) # force
    FindStyleHeaders(${search_path} READER_CLASS    reader_    READER    ) # reader    ) # read_dump
    FindStyleHeaders(${search_path} REGION_CLASS    region_    REGION    ) # region    ) # domain
endfunction(RegisterStyles)

function(RemovePackageHeader headers pkg_header)
    get_property(hlist GLOBAL PROPERTY ${headers})
    list(REMOVE_ITEM hlist ${pkg_header})
    set_property(GLOBAL PROPERTY ${headers} "${hlist}")
endfunction(RemovePackageHeader)

function(DetectAndRemovePackageHeader fname)
    RemovePackageHeader(ANGLE     ${fname})
    RemovePackageHeader(ATOM_VEC  ${fname})
    RemovePackageHeader(BODY      ${fname})
    RemovePackageHeader(BOND      ${fname})
    RemovePackageHeader(COMMAND   ${fname})
    RemovePackageHeader(COMPUTE   ${fname})
    RemovePackageHeader(DIHEDRAL  ${fname})
    RemovePackageHeader(DUMP      ${fname})
    RemovePackageHeader(FIX       ${fname})
    RemovePackageHeader(IMPROPER  ${fname})
    RemovePackageHeader(INTEGRATE ${fname})
    RemovePackageHeader(KSPACE    ${fname})
    RemovePackageHeader(MINIMIZE  ${fname})
    RemovePackageHeader(NBIN      ${fname})
    RemovePackageHeader(NPAIR     ${fname})
    RemovePackageHeader(NSTENCIL  ${fname})
    RemovePackageHeader(NTOPO     ${fname})
    RemovePackageHeader(PAIR      ${fname})
    RemovePackageHeader(READER    ${fname})
    RemovePackageHeader(REGION    ${fname})
endfunction(DetectAndRemovePackageHeader)

function(RegisterStylesExt search_path extension sources)
    FindStyleHeadersExt(${search_path} ANGLE_CLASS     ${extension}  ANGLE     ${sources})
    FindStyleHeadersExt(${search_path} ATOM_CLASS      ${extension}  ATOM_VEC  ${sources})
    FindStyleHeadersExt(${search_path} BODY_CLASS      ${extension}  BODY      ${sources})
    FindStyleHeadersExt(${search_path} BOND_CLASS      ${extension}  BOND      ${sources})
    FindStyleHeadersExt(${search_path} COMMAND_CLASS   ${extension}  COMMAND   ${sources})
    FindStyleHeadersExt(${search_path} COMPUTE_CLASS   ${extension}  COMPUTE   ${sources})
    FindStyleHeadersExt(${search_path} DIHEDRAL_CLASS  ${extension}  DIHEDRAL  ${sources})
    FindStyleHeadersExt(${search_path} DUMP_CLASS      ${extension}  DUMP      ${sources})
    FindStyleHeadersExt(${search_path} FIX_CLASS       ${extension}  FIX       ${sources})
    FindStyleHeadersExt(${search_path} IMPROPER_CLASS  ${extension}  IMPROPER  ${sources})
    FindStyleHeadersExt(${search_path} INTEGRATE_CLASS ${extension}  INTEGRATE ${sources})
    FindStyleHeadersExt(${search_path} KSPACE_CLASS    ${extension}  KSPACE    ${sources})
    FindStyleHeadersExt(${search_path} MINIMIZE_CLASS  ${extension}  MINIMIZE  ${sources})
    FindStyleHeadersExt(${search_path} NBIN_CLASS      ${extension}  NBIN      ${sources})
    FindStyleHeadersExt(${search_path} NPAIR_CLASS     ${extension}  NPAIR     ${sources})
    FindStyleHeadersExt(${search_path} NSTENCIL_CLASS  ${extension}  NSTENCIL  ${sources})
    FindStyleHeadersExt(${search_path} NTOPO_CLASS     ${extension}  NTOPO     ${sources})
    FindStyleHeadersExt(${search_path} PAIR_CLASS      ${extension}  PAIR      ${sources})
    FindStyleHeadersExt(${search_path} READER_CLASS    ${extension}  READER    ${sources})
    FindStyleHeadersExt(${search_path} REGION_CLASS    ${extension}  REGION    ${sources})
endfunction(RegisterStylesExt)

function(GenerateStyleHeaders output_path)
    GenerateStyleHeader(${output_path} ANGLE      angle     ) # force
    GenerateStyleHeader(${output_path} ATOM_VEC   atom      ) # atom      atom_vec_hybrid
    GenerateStyleHeader(${output_path} BODY       body      ) # atom_vec_body
    GenerateStyleHeader(${output_path} BOND       bond      ) # force
    GenerateStyleHeader(${output_path} COMMAND    command   ) # input
    GenerateStyleHeader(${output_path} COMPUTE    compute   ) # modify
    GenerateStyleHeader(${output_path} DIHEDRAL   dihedral  ) # force
    GenerateStyleHeader(${output_path} DUMP       dump      ) # output    write_dump
    GenerateStyleHeader(${output_path} FIX        fix       ) # modify
    GenerateStyleHeader(${output_path} IMPROPER   improper  ) # force
    GenerateStyleHeader(${output_path} INTEGRATE  integrate ) # update
    GenerateStyleHeader(${output_path} KSPACE     kspace    ) # force
    GenerateStyleHeader(${output_path} MINIMIZE   minimize  ) # update
    GenerateStyleHeader(${output_path} NBIN       nbin      ) # neighbor
    GenerateStyleHeader(${output_path} NPAIR      npair     ) # neighbor
    GenerateStyleHeader(${output_path} NSTENCIL   nstencil  ) # neighbor
    GenerateStyleHeader(${output_path} NTOPO      ntopo     ) # neighbor
    GenerateStyleHeader(${output_path} PAIR       pair      ) # force
    GenerateStyleHeader(${output_path} READER     reader    ) # read_dump
    GenerateStyleHeader(${output_path} REGION     region    ) # domain
endfunction(GenerateStyleHeaders)
