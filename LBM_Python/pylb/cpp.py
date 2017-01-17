class Cpp2d:
    def __init__(self, lattice):
        self.lattice = lattice

    def collide_boundaries(self):
        return {
            'var': ['fin', 'nx', 'ny', 'c', 't', 'q', 'wall', 'force', 'omega'],
            'code': """
        blitz::Array<double,1> feq(q);
        bool useObstacles = wall.numElements() > 0;
        bool useForce = force.numElements() > 0;

        for (int x=0; x<nx; ++x) {{
            int dy = x==0 || x==nx-1 ? 1 : ny-1;
            for (int y=0; y<ny; y+=dy) {{
                {collide_cell}
            }}
        }}
        """.format(collide_cell=self._collide_cell()) }

    def inlet_outlet(self):
        return {
            'var': ['fin', 'nx', 'ny', 'c', 't', 'q', 'bdvel',
                    'wall', 'use_inlet', 'use_outlet'],
            'code': """
        blitz::Array<double,1> feq(q);
        bool useObstacles = wall.numElements() > 0;

        for (int y=0; y<ny; ++y) {{
            int x=0;
            {inlet}
        }}

        for (int y=0; y<ny; ++y) {{
            int x=nx-1;
            {outlet}
        }}

        """.format(collide_cell=self._collide_cell(),
                   inlet=self._inlet(), outlet=self._outlet()) }

    def collide_bulk_and_stream(self):
        return {
            'var': ['fin', 'nx', 'ny', 'c', 't', 'q', 'wall', 'force', 'omega'],
            'code': """
        blitz::Array<double,1> feq(q);
        bool useObstacles = wall.numElements() > 0;
        bool useForce = force.numElements() > 0;

        for (int x=1; x<nx-1; ++x) {{
            for (int y=1; y<ny-1; ++y) {{
                {collide_cell}
                {bulkstream}
            }}
        }}
        for (int x=0; x<nx; ++x) {{
            int dy = x==0 || x==nx-1 ? 1 : ny-1;
            for (int y=0; y<ny; y+=dy) {{
                {bdstream}
            }}
        }}
        """.format(collide_cell=self._collide_cell(),
                   bulkstream=self._bulkstream(), bdstream=self._bdstream()) }

    def num_excess(self):
        return {
            'var': ['numexcess', 'nx', 'ny', 'q', 'c'],
            'code': self._excess_templ().format(bulk="""
                        numexcess(bi)++;
            """)}

    def get_excess(self):
        return {
            'var': ['fin', 'ftmp', 'ofs', 'nx', 'ny', 'q', 'c'],
            'code': self._excess_templ().format(bulk="""
                        int abs_ind = ofs(bi)+ind(bi);
                        ftmp(abs_ind) = fin(x,y, q-1-i);
                        ind(bi)++;
            """)}
            
    def put_excess(self):
        return {
            'var': ['fin', 'ftmp', 'ofs', 'nx', 'ny', 'q', 'c'],
            'code': self._excess_templ().format(bulk="""
                        if (bx==-1) xx+=nx; if (bx==+1) xx-=nx;
                        if (by==-1) yy+=ny; if (by==+1) yy-=ny;
                        int abs_ind = ofs(bi)+ind(bi);
                        fin(xx,yy, i) = ftmp(abs_ind);
                        ind(bi)++;
            """)}

    def _macroscopic(self):
        q, c = self.lattice.q, self.lattice.c
        fxy = lambda i: "fin(x,y,{0})".format(i)
        return """
            double rho = {rho};
            double u0  = ({u0_left} - ({u0_right}))/rho;
            double u1  = ({u1_left} - ({u1_right}))/rho;
        """.format(
                rho=     "+".join([fxy(i) for i in range(q)]),
                u0_left= "+".join([fxy(i) for i in range(q) if c[i,0] > 0]),
                u0_right="+".join([fxy(i) for i in range(q) if c[i,0] < 0]),
                u1_left= "+".join([fxy(i) for i in range(q) if c[i,1] > 0]),
                u1_right="+".join([fxy(i) for i in range(q) if c[i,1] < 0]))

    def _equilibrium(self):
        q, c = self.lattice.q, self.lattice.c
        def ci_dot_u(i):
            return "".join([(["-", "", "+"][c[i,d] + 1] + "u{d}").format(d=d)
                            for d in range(2) if c[i,d] != 0 ])
        eq1 = """
            double usqr = 3./2.*(u0*u0+u1*u1);
            double cu;
        """
        eq2_template = """
            cu = 3.0 * ({ci_dot_u});
            feq({pop}) = rho*t({pop})*(1.+cu+0.5*cu*cu-usqr);
        """
        eq2 = "".join([eq2_template.format(pop=i, ci_dot_u=ci_dot_u(i))
                       for i in range(q) if i != q//2 ])
        eq3 = """
            feq({i0}) = rho*t({i0})*(1.-usqr);
        """.format(i0=q//2)
        return eq1 + eq2 + eq3

    def _collision(self):
        q, c = self.lattice.q, self.lattice.c
        def ci_dot_f(i):
            return "".join([(["-", "", "+"][c[i,d] + 1] + "force(x,y,{d})").format(d=d)
                            for d in range(2) if c[i,d] != 0 ])
        add_force_template = """
             fin(x,y,{pop}) += 3.0*t({pop})*{ci_dot_f};
        """
        add_force = "".join([add_force_template.format(pop=i, ci_dot_f=ci_dot_f(i))
                             for i in range(q) if i != q//2 ])
        return """
            if (!(useObstacles && wall(x,y))) {{
                for (int i=0; i<q; ++i) {{
                    fin(x,y,i) *= 1.-omega;
                    fin(x,y,i) += omega*feq(i);
                }}
                if (useForce) {{
                    {add_force}
                }}
                for (int i=0; i<q/2; ++i) {{
                    std::swap(fin(x,y,i),fin(x,y,q-1-i));
                }}
            }}
        """.format(add_force = add_force)

    def _collide_cell(self):
        return self._macroscopic() + self._equilibrium() + self._collision()

    def _outlet(self):
        return """
            if (useObstacles && wall(x,y)) continue;
            if (use_outlet) {{
                for (int i=0; i<q; ++i) {{
                    if (c(i,0)==-1) fin(x,y,i) = fin(x-1,y,i);
                }}
            }}
        """.format(collide_cell=self._collide_cell())

    def _inlet(self):
        return  """
            if (useObstacles && wall(x,y)) continue;
            if (use_inlet) {{
                double u0 = bdvel(0,y,0);
                double u1 = bdvel(0,y,1);
                double rhoMiddle = 0., rhoLeft = 0.;
                for (int i=0; i<q; ++i) {{
                    if (c(i,0)==-1)     rhoLeft += fin(x,y,i);
                    else if (c(i,0)==0) rhoMiddle += fin(x,y,i);
                }}
                double rho = 1./(1.-u0)*(rhoMiddle+2.*rhoLeft);
                rho = 1.0;
                {equilibrium}
                for (int i=0; i<q; ++i) {{
                    if (c(i,0)==1)
                        fin(x,y,i) = fin(x,y,q-1-i) + feq(i) - feq(q-1-i);
                }}
            }}
        """.format(equilibrium=self._equilibrium(), collision=self._collision(),
                   collide_cell=self._collide_cell())

    def _bulkstream(self):
        return """
                for (int i=0; i<q/2; ++i) {{
                    int xx = x+c(i,0);
                    int yy = y+c(i,1);
                    std::swap(fin(xx,yy,i),fin(x,y,q-1-i));
                }}
        """

    def _bdstream(self):
        return """
                for (int i=0; i<q/2; ++i) {{
                    int xx = x+c(i,0);
                    int yy = y+c(i,1);
                    if (xx<0) xx=nx-1; if (xx>=nx) xx=0;
                    if (yy<0) yy=ny-1; if (yy>=ny) yy=0;
                    std::swap(fin(xx,yy,i),fin(x,y,q-1-i));
                }}
        """

    def _excess_templ(self):
        return """
        blitz::Array<size_t,1> ind(9);
        for(int i=0; i<9; ++i) ind(i)=0;
        for (int x=0; x<nx; ++x) {{
            int dy = x==0 || x==nx-1 ? 1 : ny-1;
            for (int y=0; y<ny; y+=dy) {{
                for (int i=0; i<q; ++i) {{
                    int xx = x+c(i,0); int yy = y+c(i,1);
                    int bx=0, by=0;
                    if (xx<0) bx=-1; if (xx>=nx) bx=+1;
                    if (yy<0) by=-1; if (yy>=ny) by=+1;
                    if(bx!=0 || by!=0) {{
                        int bi = by+1 + 3*(bx+1);
                        {bulk}
                    }}
                }}
            }}
        }}
        """


class Cpp3d:
    def __init__(self, lattice):
        self.lattice = lattice

    def collide_boundaries(self):
        return {
            'var': ['fin', 'nx', 'ny', 'nz', 'c', 't', 'q',
                    'wall', 'force', 'omega'],
            'code': """
        blitz::Array<double,1> feq(q);
        bool useObstacles = wall.numElements() > 0;
        bool useForce = force.numElements() > 0;

        for (int x=0; x<nx; ++x) {{
            for (int y=0; y<ny; ++y) {{
                int dz = x==0 || x==nx-1 || y==0 || y==ny-1 ? 1 : nz-1;
                for (int z=0; z<nz; z+=dz) {{
                    {collide_cell}
                }}
            }}
        }}
        """.format(collide_cell=self._collide_cell()) }

    def inlet_outlet(self):
        return {
            'var': ['fin', 'nx', 'ny', 'nz', 'c', 't', 'q', 'bdvel',
                     'wall', 'use_inlet', 'use_outlet'],
            'code': """
        blitz::Array<double,1> feq(q);
        bool useObstacles = wall.numElements() > 0;

        for (int y=0; y<ny; ++y) {{
            for (int z=0; z<nz; ++z) {{
                int x=0;
                {inlet}
            }}
        }}

        for (int y=0; y<ny; ++y) {{
            for (int z=0; z<nz; ++z) {{
                int x=nx-1;
                {outlet}
            }}
        }}

        """.format(collide_cell=self._collide_cell(),
                   inlet=self._inlet(), outlet=self._outlet()) }

    def collide_bulk_and_stream(self):
        return {
            'var': ['fin', 'nx', 'ny', 'nz', 'c', 't', 'q', 'wall', 'force', 'omega'],
            'code': """
        blitz::Array<double,1> feq(q);
        bool useObstacles = wall.numElements() > 0;
        bool useForce = force.numElements() > 0;

        for (int x=1; x<nx-1; ++x) {{
            for (int y=1; y<ny-1; ++y) {{
                for (int z=1; z<nz-1; ++z) {{
                    {collide_cell}
                    {bulkstream}
                }}
            }}
        }}
        for (int x=0; x<nx; ++x) {{
            for (int y=0; y<ny; ++y) {{
                int dz = x==0 || x==nx-1 || y==0 || y==ny-1 ? 1 : nz-1;
                for (int z=0; z<nz; z+=dz) {{
                    {bdstream}
                }}
            }}
        }}
        """.format(collide_cell=self._collide_cell(),
                   bulkstream=self._bulkstream(), bdstream=self._bdstream()) }

    def num_excess(self):
        return {
            'var':  ['numexcess', 'nx', 'ny', 'nz', 'q', 'c'],
            'code': self._excess_templ().format(bulk="""
                        numexcess(bi)++;
        """) }

    def get_excess(self):
        return {
            'var': ['fin', 'ftmp', 'ofs', 'nx', 'ny', 'nz', 'q', 'c'],
            'code': self._excess_templ().format(bulk="""
                        int abs_ind = ofs(bi)+ind(bi);
                        ftmp(abs_ind) = fin(x,y,z, q-1-i);
                        ind(bi)++;
        """) }
            
    def put_excess(self):
        return {
            'var': ['fin', 'ftmp', 'ofs', 'nx', 'ny', 'nz', 'q', 'c'],
            'code': self._excess_templ().format(bulk="""
                        if (bx==-1) xx+=nx; if (bx==+1) xx-=nx;
                        if (by==-1) yy+=ny; if (by==+1) yy-=ny;
                        if (bz==-1) zz+=nz; if (bz==+1) zz-=nz;
                        int abs_ind = ofs(bi)+ind(bi);
                        fin(xx,yy,zz, i) = ftmp(abs_ind);
                        ind(bi)++;
        """) }

    def _macroscopic(self):
        q, c = self.lattice.q, self.lattice.c
        fxy = lambda i: "fin(x,y,z,{0})".format(i)
        return """
            double rho = {rho};
            double u0  = ({u0_left} - ({u0_right}))/rho;
            double u1  = ({u1_left} - ({u1_right}))/rho;
            double u2  = ({u2_left} - ({u2_right}))/rho;
        """.format(
                rho=     "+".join([fxy(i) for i in range(q)]),
                u0_left= "+".join([fxy(i) for i in range(q) if c[i,0] > 0]),
                u0_right="+".join([fxy(i) for i in range(q) if c[i,0] < 0]),
                u1_left= "+".join([fxy(i) for i in range(q) if c[i,1] > 0]),
                u1_right="+".join([fxy(i) for i in range(q) if c[i,1] < 0]),
                u2_left= "+".join([fxy(i) for i in range(q) if c[i,2] > 0]),
                u2_right="+".join([fxy(i) for i in range(q) if c[i,2] < 0]))

    def _equilibrium(self):
        q, c = self.lattice.q, self.lattice.c
        def ci_dot_u(i):
            return "".join([(["-", "", "+"][c[i,d] + 1] + "u{d}").format(d=d)
                            for d in range(3) if c[i,d] != 0 ])
        eq1 = """
            double usqr = 3./2.*(u0*u0+u1*u1+u2*u2);
            double cu;
        """
        eq2_template = """
            cu = 3.0 * ({ci_dot_u});
            feq({pop}) = rho*t({pop})*(1.+cu+0.5*cu*cu-usqr);
        """
        eq2 = "".join([eq2_template.format(pop=i, ci_dot_u=ci_dot_u(i))
                       for i in range(q) if i != q//2 ])
        eq3 = """
            feq({i0}) = rho*t({i0})*(1.-usqr);
        """.format(i0=q//2)
        return eq1 + eq2 + eq3

    def _collision(self):
        q, c = self.lattice.q, self.lattice.c
        def ci_dot_f(i):
            return "".join([(["-", "", "+"][c[i,d] + 1] + "force(x,y,z,{d})").format(d=d)
                            for d in range(3) if c[i,d] != 0 ])
        add_force_template = """
             fin(x,y,z,{pop}) += 3.0*t({pop})*{ci_dot_f};
        """
        add_force = "".join([add_force_template.format(pop=i, ci_dot_f=ci_dot_f(i))
                             for i in range(q) if i != q//2 ])
        return """
            if (!(useObstacles && wall(x,y,z))) {{
                for (int i=0; i<q; ++i) {{
                    fin(x,y,z,i) *= 1.-omega;
                    fin(x,y,z,i) += omega*feq(i);
                }}
                if (useForce) {{
                    {add_force}
                }}
                for (int i=0; i<q/2; ++i) {{
                    std::swap(fin(x,y,z,i),fin(x,y,z,q-1-i));
                }}
            }}
        """.format(add_force = add_force)

    def _collide_cell(self):
        return self._macroscopic() + self._equilibrium() + self._collision()

    def _outlet(self):
        return """
            if (useObstacles && wall(x,y,z)) continue;
            if (use_outlet) {{
                for (int i=0; i<q; ++i) {{
                    if (c(i,0)==-1) fin(x,y,z,i) = fin(x-1,y,z,i);
                }}
            }}
        """.format(collide_cell=self._collide_cell())

    def _inlet(self):
        return  """
            if (useObstacles && wall(x,y,z)) continue;
            if (use_inlet) {{
                double u0 = bdvel(0,y,z,0);
                double u1 = bdvel(0,y,z,1);
                double u2 = bdvel(0,y,z,2);
                double rhoMiddle = 0., rhoLeft = 0.;
                for (int i=0; i<q; ++i) {{
                    if (c(i,0)==-1)     rhoLeft += fin(x,y,z,i);
                    else if (c(i,0)==0) rhoMiddle += fin(x,y,z,i);
                }}
                double rho = 1./(1.-u0)*(rhoMiddle+2.*rhoLeft);
                {equilibrium}
                for (int i=0; i<q; ++i) {{
                    if (c(i,0)==1)
                        fin(x,y,z,i) = fin(x,y,z,q-1-i) + feq(i) - feq(q-1-i);
                }}
            }}
        """.format(equilibrium=self._equilibrium(), collision=self._collision(),
                   collide_cell=self._collide_cell())

    def _bulkstream(self):
        return """
                for (int i=0; i<q/2; ++i) {{
                    int xx = x+c(i,0);
                    int yy = y+c(i,1);
                    int zz = z+c(i,2);
                    std::swap(fin(xx,yy,zz,i),fin(x,y,z,q-1-i));
                }}
        """

    def _bdstream(self):
        return """
                for (int i=0; i<q/2; ++i) {{
                    int xx = x+c(i,0);
                    int yy = y+c(i,1);
                    int zz = z+c(i,2);
                    if (xx<0) xx=nx-1; if (xx>=nx) xx=0;
                    if (yy<0) yy=ny-1; if (yy>=ny) yy=0;
                    if (zz<0) zz=nz-1; if (zz>=nz) zz=0;
                    std::swap(fin(xx,yy,zz,i),fin(x,y,z,q-1-i));
                }}
        """

    def _excess_templ(self):
        return """
        blitz::Array<size_t,1> ind(27);
        for(int i=0; i<27; ++i) ind(i)=0;
        for (int x=0; x<nx; ++x) {{
        for (int y=0; y<ny; ++y) {{
            int dz = (x==0 || x==nx-1 || y==0 || y==ny-1) ? 1 : (nz-1);
            for (int z=0; z<nz; z+=dz) {{
                for (int i=0; i<q; ++i) {{
                    int xx = x+c(i,0); int yy = y+c(i,1); int zz = z+c(i,2);
                    int bx=0, by=0, bz=0;
                    if (xx<0)   bx=-1; if (xx>=nx) bx=+1;
                    if (yy<0)   by=-1; if (yy>=ny) by=+1;
                    if (zz<0)   bz=-1; if (zz>=nz) bz=+1;
                    if(bx!=0 || by!=0 || bz!=0) {{
                        int bi = bz+1 + 3*(by+1 +3*(bx+1));
                        {bulk}
                    }}
                }}
            }}
        }}
        }}
        """
