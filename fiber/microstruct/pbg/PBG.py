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
    modes to be carried in a lower index core region.  It is also possible
    to model Photonic Crystal Fibers (PCFs) via this class using appropriate
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

        # Set Vacuum physical constants
        self.eps0 = 8.85418782e-12
        self.mu0 = 1.25663706e-6

        self.v0 = 1 / (self.eps0 * self.mu0) ** .5
        self.eta0 = np.sqrt(self.mu0 / self.eps0)

        if self.t_outer == 0:  # enforce outer region (for Modesolver)
            self.t_outer += .5 * self.r_fiber

        # Create Non-Dimensional Radii (for Modesolver)
        self.R = self.r_pml / self.scale  # beginning of PML
        self.Rout = self.r_out / self.scale  # end of PML and geometry

        # Create geometry
        self.geo = self.geometry(self.Λ, self.r_tube, self.r_fiber,
                                 self.r_poly, self.r_pml,
                                 self.r_out, self.scale, self.r_core,
                                 self.layers, self.skip, self.p, self.pattern,
                                 pml_type=self.pml_type,
                                 square_buffer=self.square_buffer)

        # Create Mesh
        self.mesh = self.create_mesh()

        # Set refinement counter
        self.refinements = 0

        # Set refractive indices (Need to implement Sellmeier here)
        self.refractive_index_dict = {'Outer': self.n_outer,
                                      'clad': self.n_clad,
                                      'tube': self.n_tube,
                                      'buffer': self.n_buffer,
                                      'core': self.n_core,
                                      'poly': self.n_poly
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
        # Sets wavelength and scaled wavelength and associated parameters
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
        self.geo = self.geometry(self.Λ, self.r_tube, self.r_fiber,
                                 self.r_poly, self.r_pml,
                                 self.r_out, self.scale, self.r_core,
                                 self.layers, self.skip, self.p,
                                 self.pattern, rot=angle)
        self.mesh = self.create_mesh()

    def create_mesh(self):
        """Set materials, max diameters and create mesh."""
        # Set the materials for the domain.
        mat = {6: 'poly', 5: 'Outer', 4: 'buffer',
               3: 'tube', 2: 'clad', 1: 'core'}

        for domain, material in mat.items():
            self.geo.SetMaterial(domain, material)

        # Set the maximum mesh sizes in subdomains

        self.geo.SetDomainMaxH(1, self.core_maxh)
        self.geo.SetDomainMaxH(2, self.clad_maxh)
        self.geo.SetDomainMaxH(3, self.tube_maxh)
        self.geo.SetDomainMaxH(4, self.buffer_maxh)
        self.geo.SetDomainMaxH(5, self.pml_maxh)
        self.geo.SetDomainMaxH(6, self.poly_maxh)

        print("Generating mesh.")
        mesh = ng.Mesh(self.geo.GenerateMesh())
        print("Mesh created.")

        mesh.Curve(3)

        return mesh

    def reset_mesh(self):
        """Reset to original mesh."""
        self.geo = self.geometry(self.Λ, self.r_tube, self.r_fiber,
                                 self.r_poly, self.r_pml,
                                 self.r_out, self.scale, self.r_core,
                                 self.layers, self.skip, self.p,
                                 self.pattern, pml_type=self.pml_type,
                                 square_buffer=self.square_buffer)

        self.mesh = self.create_mesh()

    def geometry(self, Λ, r_tube, r_fiber, r_poly, r_pml, r_out, scale, r_core,
                 layers=6, skip=1, p=6, pattern=[], rot=0, hexcore=True,
                 pml_type='radial', square_buffer=.5):
        """
        Construct and return Non-Dimensionalized geometry.

        Parameters
        ----------
        Λ : float
            Distance between layers of tubes (Dimensional).
        r : float
            Radius of tubes (Dimensional).
        r_fiber : int or float
            Radius of fiber (Dimensional).
        r_poly: int or float
            Radius at which to begin polymer
        r_pml : int or float
            Radius at which to begin PML (Dimensional).
        r_out : int or float
            Radius at which to end PML (Dimensional).
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
        rot: float, optional
            Rotate mesh by 'rot' radians. The default is zero.
        hexcore: boolean, optional
            Create hexagonal core region.  If set to false a circular core
            region is created.  For a hexagonal core, the radius is defined
            as the distance from the center to a vertex of the hexagon, and
            the value used for this is r_core. The default is True.
        pml_type: string, optional
            Determine geometrical type of pml. Options are 'radial' or
            'square'. If square is chosen, buffer region must be included, and
            this is ensured using the square_buffer parameter.  Default is
            'radial'.
        square_buffer: float, optional
            Enforce buffer thickness for square pml.  If provided radius of
            fiber and radius of buffer are equal, this will add a square
            buffer region with minimal distance to the fiber of
            square_buffer * R_fiber.  The default of .5 ensures buffer region
            is at least half a fiber radius separated from the fiber itself.

        Returns
        -------
        geo : netgen.libngpy._geom2d.SplineGeometry
            Completed geometry.
        """
        geo = geom2d.SplineGeometry()

        # Non-Dimensionalize needed physical parameters
        Λ /= scale
        R_tube = r_tube / scale

        R_fiber = r_fiber / scale
        R_poly = r_poly / scale
        R_pml = r_pml / scale
        R_out = r_out / scale

        # Create core region
        if skip == 0 or r_core == 0:
            pass

        elif hexcore:
            R_core = r_core / scale
            coords = [(R_core * np.cos(i * 2 * np.pi / p),
                       R_core * np.sin(i * 2 * np.pi / p))
                      for i in range(p)]
            pts = [geo.AppendPoint(x, y) for x, y in coords]

            for i in range(p - 1):
                geo.Append(["line", pts[i], pts[i+1]], leftdomain=1,
                           rightdomain=2)

            geo.Append(["line", pts[-1], pts[0]], leftdomain=1,
                       rightdomain=2)

        else:
            R_core = r_core / scale
            geo.AddCircle(c=(0, 0), r=R_core, leftdomain=1, rightdomain=2,
                          bc='computational_core_cladding_interface')

        # Add the layers of tubes
        for i in range(layers):

            index_layer = skip + i
            Ri = index_layer * Λ

            if len(pattern) > 0:
                self.add_layer(geo, R_tube, Ri, p=p, innerpoints=index_layer-1,
                               mask=pattern[i], rot=rot)
            else:
                self.add_layer(geo, R_tube, Ri, p=p, innerpoints=index_layer-1,
                               rot=rot)

        # Create boundary of fiber, polymer and PML region

        if pml_type == 'radial':

            if R_fiber == R_pml:  # No buffer layer or polymer layer
                print('no buffer no polymer')
                geo.AddCircle(c=(0, 0), r=R_pml, leftdomain=2, rightdomain=5,
                              bc='fiber_pml_interface')
                geo.AddCircle(c=(0, 0), r=R_out,
                              leftdomain=5, bc="OuterCircle")

            else:  # one or both exist
                if R_fiber == R_poly:  # No polymer layer, but yes buffer

                    geo.AddCircle(c=(0, 0), r=R_fiber, leftdomain=2,
                                  rightdomain=4, bc='fiber_buffer_interface')
                    geo.AddCircle(c=(0, 0), r=R_pml, leftdomain=4,
                                  rightdomain=5, bc='buffer_pml_interface')
                    geo.AddCircle(c=(0, 0), r=R_out,
                                  leftdomain=5, bc="OuterCircle")

                elif R_poly == R_pml:  # No buffer layer, but yes polymer

                    geo.AddCircle(c=(0, 0), r=R_fiber, leftdomain=2,
                                  rightdomain=6, bc='fiber_polymer_interface')
                    geo.AddCircle(c=(0, 0), r=R_poly, leftdomain=6,
                                  rightdomain=5, bc='polymer_pml_interface')
                    geo.AddCircle(c=(0, 0), r=R_out,
                                  leftdomain=5, bc="OuterCircle")

                else:  # Both polymer and buffer
                    geo.AddCircle(c=(0, 0), r=R_fiber, leftdomain=2,
                                  rightdomain=6, bc='fiber_polymer_interface')
                    geo.AddCircle(c=(0, 0), r=R_poly, leftdomain=6,
                                  rightdomain=4, bc='polymer_buffer_interface')
                    geo.AddCircle(c=(0, 0), r=R_pml, leftdomain=4,
                                  rightdomain=5, bc='buffer_pml_interface')
                    geo.AddCircle(c=(0, 0), r=R_out,
                                  leftdomain=5, bc="OuterCircle")

        elif pml_type == 'square':

            if R_poly == R_pml:  # Need to enforce buffer space
                R_pml = (1 + square_buffer) * R_poly  # give buffer space

            self.r_pml_square = R_pml * scale  # add as attribute
            self.R_pml_square = R_pml
            if R_fiber == R_poly:  # no polymer layer, just buffer

                geo.AddCircle(c=(0, 0), r=R_fiber, leftdomain=2, rightdomain=4,
                              bc='fiber_buffer_interface')
                geo.AddRectangle((-R_pml, -R_pml), (R_pml, R_pml),
                                 leftdomain=4, rightdomain=5,
                                 bc='buffer_pml_interface')
                geo.AddRectangle((-R_out, -R_out), (R_out, R_out),
                                 bc='OuterCircle',
                                 leftdomain=5)

            else:  # must be buffer layer, so we have polymer and buffer

                geo.AddCircle(c=(0, 0), r=R_fiber, leftdomain=2, rightdomain=6,
                              bc='fiber_polymer_interface')
                geo.AddCircle(c=(0, 0), r=R_poly, leftdomain=6, rightdomain=4,
                              bc='polymer_buffer_interface')
                geo.AddRectangle((-R_pml, -R_pml), (R_pml, R_pml),
                                 leftdomain=4, rightdomain=5,
                                 bc='buffer_pml_interface')
                geo.AddRectangle((-R_out, -R_out), (R_out, R_out),
                                 leftdomain=5, bc='OuterCircle')
        else:
            raise NotImplementedError('PML type must be square or radial')

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
            geo.AddCircle(c=(0, 0), r=r, leftdomain=3, rightdomain=2,
                          bc='core_cladding_interface')

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
                geo.AddCircle(c=(x, y), r=r, leftdomain=3,
                              rightdomain=2, bc='microtube_cladding_interface')

    def refine(self, n=1):
        """Refine mesh n times."""
        self.refinements += n
        for i in range(n):
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
