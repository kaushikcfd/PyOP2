import loopy as lp
from typing import List
from dataclasses import dataclass


@dataclass
class DQInference:
    iname_trial_dof_init: str
    iname_out_dof_init: str
    iname_out_dof_scatter: str
    var_trial_dofs: List[str]
    var_out_dof: str
    var_quadrature_temps: List[str]
    iname_quadrature_init: str
    var_pre_eval_acc: List[str]
    iname_pre_eval_row: str
    var_quad_wrap_up_result: List[str]
    iname_quad_eval_quadr: str
    insn_eval: List[str]
    iname_quadr_idof_init: str
    iname_quadr_idof: str
    iname_quadr_idof_wrapup: str
    var_eval_result: str
    iname_eval_wrap_up_row: str
    insn_quadr: List[str]


inferred_operators = {
        "mass": DQInference(iname_trial_dof_init="i0",
                            iname_out_dof_init="i3",
                            iname_out_dof_scatter="i4",
                            var_trial_dofs=["t0"],
                            var_out_dof="t2",
                            var_quadrature_temps=["form_t13", "form_t14"],
                            iname_quadrature_init="form_ip1",
                            var_pre_eval_acc=["form_t15"],
                            iname_pre_eval_row="form_i",
                            var_quad_wrap_up_result=["form_t19"],
                            iname_quad_eval_quadr="form_ip1_0",
                            insn_eval=["form_insn_16", "form_insn_17", "form_insn_18"],
                            iname_quadr_idof_init="form_j1",
                            iname_quadr_idof="form_j1_0",
                            iname_quadr_idof_wrapup="form_j1_1",
                            var_eval_result="form_t20",
                            iname_eval_wrap_up_row="form_j0",
                            insn_quadr=["form_insn_19"],
                            ),
        }


def dq_transform(kernel, nc, nwi):
    ndof_1d = int(lp.symbolic.pw_aff_to_expr(kernel.get_iname_bounds("form_i", constants_only=True).size))
    print(f"ndof_1d is {ndof_1d}, Nwi is {nwi}, Nc is {nc}.")

    knlinfo = inferred_operators["mass"]

    # {{{ remove noops

    noop_insns = set([insn.id
                      for insn in kernel.instructions
                      if isinstance(insn, lp.NoOpInstruction)])
    kernel = lp.remove_instructions(kernel, noop_insns)

    from loopy.transform.instruction import remove_unnecessary_deps
    kernel = remove_unnecessary_deps(kernel)

    # }}}

    from loopy.transform.make_scalar import remove_axis
    from loopy.transform.data import remove_unused_axes_in_temporaries
    kernel = remove_unused_axes_in_temporaries(kernel)

    #  splitting t0 and t2 accesses to make them less ugly.
    kernel = lp.split_iname(kernel, knlinfo.iname_trial_dof_init, ndof_1d, outer_tag="unr")
    kernel = lp.split_iname(kernel, knlinfo.iname_out_dof_init, ndof_1d, outer_tag="unr")
    kernel = lp.split_iname(kernel, knlinfo.iname_out_dof_scatter, ndof_1d, outer_tag="unr")
    kernel = lp.split_iname(kernel, f"{knlinfo.iname_trial_dof_init}_inner", nwi, inner_tag="l.0")
    assert len(knlinfo.var_trial_dofs) == 1
    kernel = lp.split_array_axis(kernel, knlinfo.var_trial_dofs[0], 0, ndof_1d)
    kernel = lp.split_array_axis(kernel, knlinfo.var_out_dof, 0, ndof_1d)

    # Distributing work to the SMs.
    kernel = lp.split_iname(kernel, "n", nc, inner_iname="icell", outer_tag="g.0", inner_tag="l.1")

    # Stage1: Storing the per-quadrature point data.
    kernel = lp.privatize_temporaries_with_inames(kernel, "icell", knlinfo.var_quadrature_temps)
    kernel = lp.set_temporary_scope(kernel, knlinfo.var_quadrature_temps, "local")
    kernel = lp.split_iname(kernel, knlinfo.iname_quadrature_init, nwi, inner_tag="l.0")

    # Stage 2: The first set parallel inner products
    kernel = lp.privatize_temporaries_with_inames(kernel, "icell", knlinfo.var_pre_eval_acc)
    kernel = lp.set_temporary_scope(kernel, knlinfo.var_pre_eval_acc, "local")
    kernel = lp.split_iname(kernel, knlinfo.iname_pre_eval_row, nwi, inner_tag="l.0", outer_tag="unr")
    kernel = lp.split_array_axis(kernel, knlinfo.var_trial_dofs[0], 1, nwi)
    kernel = remove_axis(kernel, knlinfo.var_trial_dofs[0], 2)

    # Start of the matvec after matvec stage.
    # Stage 3.1: The stage which seems like the evaluation stage of the matvec-after-matvec stage.
    kernel = lp.privatize_temporaries_with_inames(kernel, "icell", knlinfo.var_quad_wrap_up_result)
    kernel = lp.privatize_temporaries_with_inames(kernel, knlinfo.iname_quad_eval_quadr, knlinfo.var_quad_wrap_up_result)
    kernel = lp.duplicate_inames(kernel, knlinfo.iname_quad_eval_quadr, " or ".join([f"id:{k}" for k in knlinfo.insn_eval]), f"{knlinfo.iname_quad_eval_quadr}_eval")
    kernel = lp.set_temporary_scope(kernel, knlinfo.var_quad_wrap_up_result, "local")
    kernel = lp.split_iname(kernel, f"{knlinfo.iname_quad_eval_quadr}_eval", nwi, inner_tag="l.0", outer_tag="unr")

    # Stage 3.2: The stage which seems like the quadrature stage of the matvec-after-matvec stage.
    kernel = lp.duplicate_inames(kernel, f"{knlinfo.iname_quad_eval_quadr}", " or ".join([f"id:{k}" for k in knlinfo.insn_quadr]), f"{knlinfo.iname_quad_eval_quadr}_quadr")
    kernel = lp.rename_iname(kernel, knlinfo.iname_quadr_idof, knlinfo.iname_quadr_idof_init, existing_ok=True)
    kernel = lp.rename_iname(kernel, knlinfo.iname_quadr_idof_wrapup, knlinfo.iname_quadr_idof_init, existing_ok=True)
    kernel = lp.split_iname(kernel, knlinfo.iname_quadr_idof_init, nwi, inner_tag="l.0", outer_tag="unr")
    kernel = remove_axis(kernel, knlinfo.var_eval_result, 0)
    kernel = lp.split_iname(kernel, f"{knlinfo.iname_out_dof_init}_inner", nwi, inner_tag="l.0", outer_tag="unr")
    kernel = lp.split_iname(kernel, f"{knlinfo.iname_out_dof_scatter}_inner", nwi, inner_tag="l.0", outer_tag="unr")
    kernel = lp.split_array_axis(kernel, knlinfo.var_out_dof, 1, nwi)
    kernel = remove_axis(kernel, knlinfo.var_out_dof, 2)

    kernel = lp.tag_inames(kernel, f"{knlinfo.iname_eval_wrap_up_row}:unr")

    return kernel, ()
