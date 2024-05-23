import csv
import json
import quanscient as qs
from utils import Mesh, Variables, Empty, Fields, DerivedFields
from expressions import expr
from materials import mat
from regions import reg
from parameters import par

var = Variables()
mesh = Mesh()
fld = Fields()
df = DerivedFields()

# Load the mesh
mesh.mesh = qs.mesh()
mesh.mesh.setphysicalregions(*reg.get_region_data())
mesh.skin = reg.get_next_free()
mesh.mesh.selectskin(mesh.skin)
mesh.mesh.partition()
mesh.mesh.load("gmsh:simulation.msh", mesh.skin, 1, 1)

# Magnetic scalar potential field
fld.phi = qs.field("h1", [1])
fld.phi.setorder(reg.all, 2)

df.H = qs.parameter(3, 1)
df.B = qs.parameter(3, 1)

# Magnetic field
df.H.addvalue(reg.all, -qs.grad(fld.phi))

# Magnetic flux density
df.B.addvalue(reg.all, -par.mu() * qs.grad(fld.phi))
df.B.addvalue(reg.remanence_n55_y__target_2, qs.array3x1(0.0, (-1.51), 0.0))
df.B.addvalue(reg.remanence_n55_z__target_2, qs.array3x1(0.0, 0.0, (-1.51)))
df.B.addvalue(reg.remanence_0_target, qs.array3x1(0.0, 0.0, 0.0))
df.B.addvalue(reg.remanence_n55_z__target, qs.array3x1(0.0, 0.0, 1.51))
df.B.addvalue(reg.remanence_n55_y__target, qs.array3x1(0.0, 1.51, 0.0))

form = qs.formulation()

# Magnetism φ
form += qs.integral(reg.all, -par.mu() * qs.grad(qs.dof(fld.phi)) * qs.grad(qs.tf(fld.phi)))

# Remanence interaction: Remanence_N55_Z
form += qs.integral(reg.remanence_n55_z__target, qs.array3x1(0.0, 0.0, 1.51) * qs.grad(qs.tf(fld.phi)))

# Remanence interaction: Remanence_N55_Z-
form += qs.integral(reg.remanence_n55_z__target_2, qs.array3x1(0.0, 0.0, (-1.51)) * qs.grad(qs.tf(fld.phi)))

# Remanence interaction: Remanence_N55_Y
form += qs.integral(reg.remanence_n55_y__target, qs.array3x1(0.0, 1.51, 0.0) * qs.grad(qs.tf(fld.phi)))

# Remanence interaction: Remanence_N55_Y-
form += qs.integral(reg.remanence_n55_y__target_2, qs.array3x1(0.0, (-1.51), 0.0) * qs.grad(qs.tf(fld.phi)))

# Remanence interaction: Remanence_0
form += qs.integral(reg.remanence_0_target, qs.array3x1(0.0, 0.0, 0.0) * qs.grad(qs.tf(fld.phi)))

form.allsolve(relrestol=1e-06, maxnumit=1000, nltol=1e-05, maxnumnlit=-1, relaxvalue=-1)

# Field output: B Air Outer
qs.setoutputfield("B Air Outer", reg.b_air_outer_target, df.B, 2)

# Field output: B Magnets
qs.setoutputfield("B Magnets", reg.b_magnets_target, df.B, 2)

# Field output: B Air Inner
qs.setoutputfield("B Air Inner", reg.b_air_inner_target, df.B, 2)
