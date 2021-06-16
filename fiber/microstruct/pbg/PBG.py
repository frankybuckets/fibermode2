import netgen.geom2d as geom2d
import ngsolve as ng
import numpy as np
import os
import pickle
from fiberamp.fiber.modesolver import ModeSolver
from pyeigfeast.spectralproj.ngs import NGvecs


class PBG(ModeSolver):
    """
    Create a Photonic Band Gap (PBG) fiber object.

    These types of fibers have a lattice like microstructure that can allow
    for modes to be carried in a lower index core region.  It is also possible
    to model Photonic Crystal Fibers (PCFs) using this class using appropriate
    parameters.  The PBG object can then be used to find the modes of the
    associated fiber using methods from the parent class ModeSolver.

    Inputs for Constructor
    ----------
    fiber_param_dict : dict
        Dictionary of parameters needed to construct the fiber. These are
        set as attributes.

    Attributes
    ----------
    - p: int
        The number of sides of the polygonal lattice. Default is 6.
    - layers: int
        Number of layers of tubes forming microstructure region.
    - skip: int
        Number of layers skipped to make the core region. Does not subtract
        from total number of tubes forming microstructure region.
    - Λ: float
        Distance separating layers of tubes.
    - r_tube: float
        Radius of the tubes.
    - r_core: float
        Radius of the core region (formed by skipping layers).
    - r_fiber: float
        Radius of the fiber as a whole.  More generally this can be
        any radius after which refractive profile is homogeneous.
    - scale: float
        Factor by which to scale the fiber parameters to make a
        non-dimensional geometry.  Frequently chosen to make the core
        region unit radius.  Note: resetting scale does NOT reset attributes
        derived from it (including geo, mesh, V).  If you change the scale you
        need to rebuild the object.
    - n_tube, n_clad: float
        Refractive indices of the tube and cladding material respectively.
    - n_core, n_buffer, n_outer: float
        Refractive indices in the respective regions.
    - n0 : float
        Base refractive index used in the refractive index function V.
    - t_buffer, t_outer: float
        Thickness of the buffer and PML regions respectively.
    - alpha: float
        PML parameter.
    - pml_maxh, air_maxh, tube_maxh, clad_maxh, core_maxh: floats
        Maximum element diameter for mesh on respective regions.
    - R_fiber: float
        Non-Dimensional radius after which refractive index function is
        constant.
    - R, Rout: floats
        Non-Dimensional radii indicating the beginning and end of the PML
        region.
    - wavelength, k: floats
        Wavelength and wavenumber of light for which we seek modes. Setting
        wavelength automatically sets k and V (see below).
    - geo, mesh: NGsolve objects
        Geometry and mesh of fiber.
    - refractive_index_dict: dict
        Dictionary giving refractive index of fiber materials.
    - V: NGsolve CoefficientFunction object
        Coefficient function based on refractive index function N, used in
        differential equation to be solved.
    - N: NGsolve CoefficientFunction object
        The refractive index function of the fiber. Often referred to as 'n'
        in the literature. By default this is a piecewise constant function
        equal to the refractive index of the material. Can be updated to any
        desired coefficient function defined on the mesh materials.  To reset
        to original values use self.reset_N method.  Note: updating N updates
        V function.

    Methods
    -------
    See parent class ModeSolver.

    """

    def __init__(self, fiber_param_dict):

        for key, value in fiber_param_dict.items():
            if key == 'wavelength':  # set later to update k and V
                pass
            else:
                setattr(self, key, value)

        # Create Non-Dimensional Radii
        self.R_fiber = self.r_fiber / self.scale
        self.R = (self.r_fiber + self.t_buffer) / self.scale
        self.Rout = (self.r_fiber + self.t_buffer + self.t_outer) / self.scale

        # Create geometry
        self.geo = self.geometry(self.Λ, self.r_tube, self.R_fiber, self.R,
                                 self.Rout, self.scale, self.r_core,
                                 self.layers, self.skip, self.p, self.pattern)

        # Create Mesh
        self.mesh = self.create_mesh()
        self.refinements = 0
        self.refractive_index_dict = {'Outer': self.n_outer,
                                      'clad': self.n_clad,
                                      'tube': self.n_tube,
                                      'buffer': self.n_buffer,
                                      'core': self.n_core
                                      }

        # Set wavelength and base coefficent function (these then set k and V)
        self.wavelength = fiber_param_dict['wavelength']
        self.N = self.refractive_index_dict

        # Initialize parent class ModeSolver
        super().__init__(self.mesh, self.scale, self.n0)

    @property
    def wavelength(self):
        """Get wavelength."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, lam):
        self._wavelength = lam
        self.k = 2 * np.pi / self._wavelength
        try:
            self.V = self.set_V(self.N, self.k)
        except AttributeError:
            pass

    @property
    def N(self):
        """Get N."""
        return self._N

    @N.setter
    def N(self, ref_coeff_info):
        """Set base refractive coefficient function."""
        if type(ref_coeff_info) == dict:
            mats = self.mesh.GetMaterials()
            self._N = ng.CoefficientFunction(
                [ref_coeff_info[mat] for mat in mats])

        elif type(ref_coeff_info) == ng.CoefficientFunction:
            self._N = ref_coeff_info

        else:
            raise NotImplementedError("Only dictionaries or coefficient\
                                      functions can be used to set base\
                                    refractive index function N.")
        self.V = self.set_V(self._N, self.k)

    def set_V(self, N, k):
        """Set coefficient function (V) for mesh."""
        V = (self.scale * k) ** 2 * (self.n0 ** 2 - N ** 2)
        return V

    def reset_N(self):
        """Reset N to piecewise constant refractive index function."""
        self.N = self.refractive_index_dict

    def rotate(self, angle):
        """Rotate fiber by 'angle' (radians)."""
        self.geo = self.geometry(self.Λ, self.r_tube, self.R_fiber, self.R,
                                 self.Rout, self.scale, self.r_core,
                                 self.layers, self.skip, self.p,
                                 self.pattern, rot=angle)
        self.mesh = self.create_mesh()

    def create_mesh(self):
        """Set materials, max diameters and create mesh."""
        # Set the materials for the domain.
        mat = {5: 'Outer', 4: 'buffer', 3: 'tube', 2: 'clad', 1: 'core'}

        for domain, material in mat.items():
            self.geo.SetMaterial(domain, material)

        # Set the maximum mesh sizes in subdomains

        self.geo.SetDomainMaxH(1, self.core_maxh)
        self.geo.SetDomainMaxH(2, self.clad_maxh)
        self.geo.SetDomainMaxH(3, self.tube_maxh)
        self.geo.SetDomainMaxH(4, self.buffer_maxh)
        self.geo.SetDomainMaxH(5, self.pml_maxh)

        print("Generating mesh.")
        mesh = ng.Mesh(self.geo.GenerateMesh())
        print("Mesh created.")

        mesh.Curve(3)

        return mesh

    def reset_mesh(self):
        """Reset to original mesh."""
        self.geo = self.geometry(self.Λ, self.r_tube, self.R_fiber, self.R,
                                 self.Rout, self.scale, self.r_core,
                                 self.layers, self.skip, self.p,
                                 self.pattern)
        self.mesh = self.create_mesh()

    def geometry(self, Λ, r, R_fiber, R, Rout, scale, r_core, layers=6,
                 skip=1, p=6, pattern=[], rot=0):
        """
        Construct and return Non-Dimensionalized geometry.

        Parameters
        ----------
        Λ : float
            Distance between layers of tubes.
        r : float
            Radius of tubes.
        R_fiber : int or float
            Radius of fiber (non-dimensional).
        R : int or float
            Radius at which to begin PML (non-dimensional).
        Rout : int or float
            Radius at which to end PML (non-dimensional).
        scale : float, optional
            Physical scale factor.
        layers : int, optional
            Number of layers of tubes. The default is 6.
        skip : int, optional
            Number of layers to skip to create core region. Note that this \
            does not reduce the number of layers created as determined by\
            the variable "layers." The default is 1.
        p : int, optional
            Number of sides of polygon determining lattice. The default is 6.
        pattern : list, optional
            List of 0's and 1's determining pattern of tubes. The default is
            [].

        Returns
        -------
        geo : netgen.libngpy._geom2d.SplineGeometry
            Completed geometry.
        """
        geo = geom2d.SplineGeometry()

        # Non-Dimensionalize needed physical parameters
        Λ /= scale
        r /= scale

        # Create core region
        if skip == 0 or r_core == 0:
            pass

        else:
            R_core = r_core / scale
            geo.AddCircle(c=(0, 0), r=R_core, leftdomain=1, rightdomain=2)

        # Add the layers of tubes
        for i in range(layers):

            index_layer = skip + i
            Ri = index_layer * Λ

            if len(pattern) > 0:
                self.add_layer(geo, r, Ri, p=p, innerpoints=index_layer - 1,
                               mask=pattern[i], rot=rot)
            else:
                self.add_layer(geo, r, Ri, p=p, innerpoints=index_layer - 1,
                               rot=rot)

        # Create boundary of fiber and PML region

        if R_fiber == R:

            # No buffer layer
            geo.AddCircle(c=(0, 0), r=R, leftdomain=2, rightdomain=5)
            geo.AddCircle(c=(0, 0), r=Rout, leftdomain=5,
                          bc="OuterCircle")  # outermost circle

        else:

            # Create buffer layer
            geo.AddCircle(c=(0, 0), r=R_fiber, leftdomain=2, rightdomain=4)
            geo.AddCircle(c=(0, 0), r=R, leftdomain=4, rightdomain=5)
            geo.AddCircle(c=(0, 0), r=Rout, leftdomain=5, bc="OuterCircle")

        return geo

    def add_layer(self, geo, r, R, p=6, innerpoints=0, mask=None, rot=0):
        """
        Add a single layer of small circles of radius r at vertices of p-sided\
        polygon (inscribed in circle of radius R) to geometry geo.

        If innerpoints is greater than zero, also adds circles along edges of
        polygon.

        Parameters
        ----------
        geo : netgen.libngpy._geom2d.SplineGeometry
            Geometry to which to add layer of circles.
        r : float
            Small circle radius.
        R : float
            Large circle radius.
        p : int, optional
            Number of sides of polygon. The default is 6.
        innerpoints : int, optional
            Number of points on each edge (excluding vertices). The default
            is 0.
        mask : list, optional
            List of zeros and ones. Used to form pattern. The default is None.
        rot: float, optional
            Rotate polygon by 'rot' radians. The default is 0.

        Raises
        ------
        ValueError
            Number of polygon sides must be at least 2.

        """
        if p == 0 or p == 1:
            raise ValueError("Please specify a p of 2 or greater.")

        if R == 0:         # zero big radius draws circle at origin
            geo.AddCircle(c=(0, 0), r=r, leftdomain=3, rightdomain=2)

        else:

            # Create lists of x and y coordinates
            xs = []
            ys = []

            for i in range(p):

                # Get vertex points of polygon

                x0, y0 = R * np.cos(i * 2 * np.pi / p + rot), R * \
                    np.sin(i * 2 * np.pi / p + rot)
                x1, y1 = R * np.cos((i + 1) * 2 * np.pi / p + rot), R * \
                    np.sin((i + 1) * 2 * np.pi / p + rot)

                for s in range(innerpoints + 1):

                    # Interpolate between vertices to get innerpoints

                    t = s * 1 / (innerpoints + 1)
                    xs.append((1 - t) * x0 + t * x1)
                    ys.append((1 - t) * y0 + t * y1)

            if mask is not None:

                # Mask out unwanted points
                xs = list(np.array(xs)[np.where(mask)])
                ys = list(np.array(ys)[np.where(mask)])

            for x, y in zip(xs, ys):

                # Add the circles
                geo.AddCircle(c=(x, y), r=r, leftdomain=3, rightdomain=2)

    def ndofs(self, p, refs):
        """Find number of dofs of fes (order=p) of mesh (with\
        refinements=refs)."""
        # Find number of edges elts and verts after refinements
        edges = self.mesh.nedge
        elts = self.mesh.ne
        verts = self.mesh.nv

        for r in range(refs):
            verts = verts + edges
            edges = 2 * edges + 3 * elts
            elts = 4 * elts

        # Total grid points for poly is p + d choose d so (p + 2) * (p + 1) / 2
        # strictly interior points is given by (p - 1) * (p - 2) / 2.
        interior_points = (p - 1) * (p - 2) * elts / 2

        # edge interiors contribute p-1 each
        edge_interior_points = (p - 1) * edges

        # each vert gives one too, so in total:
        return verts + interior_points + edge_interior_points

    def refine(self):
        """Refine mesh by dividing each triangle into four."""
        self.refinements += 1
        self.mesh.ngmesh.Refine()
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())
        self.mesh.Curve(3)

    # SAVE & LOAD #####################################################

    def save(self, name, folder):
        """Save PBG object.

        Parameters
        ----------
        name : str
            Desired file name.  The suffix '_pbg.pkl' will be attached.
        folder : str
            Path to destination folder.  May be absolute or relative to current
            directory.  Exception is raised if folder does not exist.
        """
        filename = os.path.relpath(folder + '/' + name + '_pbg.pkl')
        if os.path.isdir(folder):

            print('Pickling pbg object into ', filename)
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise OSError("The given folder is not a directory.\
                          Please check and make directory as needed.")

    def savemodes(self, name, folder, Y, p, betas, Zs, solverparams=None,
                  longY=None, longYl=None, pbgpickle=False):
        """
        Save an NGVecs span object Y containing modes of FE degree p.

        Include any solver parameters to be saved together with the
        modes in the input dictionary "solverparams". If "pbgpickle"
        is True, then the pbg object is also saved under the same "name".


        Parameters
        ----------
        name : str
            Desired file name.  The suffix '_mode.npz' will be attached.
        folder : str
            Path to destination folder.  May be absolute or relative to current
            directory.  Exception is raised if folder does not exist.
        Y : NGvecs object
            Object containing modes to be saved.
        p : int
            Finite element degree of associated modes.
        betas : str
            Propagation constants of associated modes.
        Zs : ndarray
            Eigenvalues of associated modes.
        solverparams : dict
            Extra solver parameters to save. The default is None.
        longY : ndarray, optional
            Long eigenvectors from linearized problem. The default is None.
        longYl : ndarray, optional
            Long left eigenvectors from linearized problem. The default is
            None.
        pbgpickle : boolean, optional
            If set to True, this function will also save the associated PBG
            object under the same name (with suffix '_npg.pkl' attached).
            The default is False.

        Returns
        -------
        None.

        """
        if pbgpickle:
            self.save(name, folder)
        y = Y.tonumpy()
        if longY is not None:
            longY = longY.tonumpy()
        if longYl is not None:
            longYl = longYl.tonumpy()
        d = {'y': y, 'p': p, 'betas': betas, 'Zs': Zs,
             'longy': longY, 'longyl': longYl}
        if solverparams is not None:
            d.update(**solverparams)

        f = os.path.relpath(folder + '/' + name + '_mode.npz')
        print('Writing mode file ', f)
        np.savez(f, **d)

# End of class PBG ###################################


def load_pbg(name, folder=''):
    """Load a saved pbg object from file <name>_pbg.pkl in <folder>."""
    pbgfile = os.path.relpath(folder + '/' + name + '_pbg.pkl')
    with open(pbgfile, 'rb') as f:
        a = pickle.load(f)
    return a


def load_pbg_mode(mode_prefix, pbg_prefix, mode_folder='', pbg_folder=''):
    """

    Load a mode saved in npz file <mode_prefix> compatible with the pbg object\
    saved in pickle file <pbg_prefix>_pbg.pkl.

    Parameters
    ----------
    mode_prefix : str
        Name of mode file.
    pbg_prefix : str
        Name of PBG object file.
    mode_folder : str, optional
        Location of folder containing modes.
    pbg_folder : str, optional
        Location of folder containing PBG object.

    Returns
    -------
    a : PBG object
        PBG object associated with loaded modes.
    Y : NGvecs
        Modes.
    betas : ndarray
        Propagation constants associated with modes.
    Zs : ndarray
        Eigenvalues associated with modes.
    p : int
        Degree of associated modes.
    d : dict
        Dictionary containing extra solver parameters..
    longY : NGvecs
        Long eigenvectors from linearized problem..
    longYl : NGvecs
        Long left eigenvectors from linearized problem..

    """
    a = load_pbg(pbg_prefix, pbg_folder)   # load PBG object

    mode_path = os.path.relpath(mode_folder + '/' + mode_prefix + '_mode.npz')
    d = dict(np.load(mode_path, allow_pickle=True))  # load saved dictionary

    # Extract entries
    p = int(d.pop('p'))
    betas = d.pop('betas')
    Zs = d.pop('Zs')
    y = d.pop('y')

    for k, v in d.items():
        if v.ndim > 0:
            d[k] = v
        else:  # convert singleton arrays to scalars
            d[k] = v.item()

    # Reconstruct NGvecs and set their data
    X = ng.H1(a.mesh, order=p, complex=True)
    X3 = ng.FESpace([X, X, X])
    Y = NGvecs(X, y.shape[1])
    Y.fromnumpy(y)
    longY = None
    longYl = None
    longy = d['longy']
    longyl = d['longyl']

    if longy is not None:
        longY = NGvecs(X3, longy.shape[1])
        longY.fromnumpy(longy)
    if longyl is not None:
        longYl = NGvecs(X3, longyl.shape[1])
        longYl.fromnumpy(longyl)

    return a, Y, betas, Zs, p, d, longY, longYl
