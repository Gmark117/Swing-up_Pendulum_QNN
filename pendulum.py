'''
Create a simulation environment for a N-pendulum.
Example of use:

env = Pendulum(N)
env.reset()

for i in range(1000):
   env.step(zero(env.nu))
   env.render()

'''

import numpy as np
import pinocchio as pin
from display import Display
from numpy.linalg import inv
import time


class Visual:
    '''
    Class representing one 3D mesh of the robot, to be attached to a joint. The class contains:
    * the name of the 3D objects inside Gepetto viewer.
    * the ID of the joint in the kinematic tree to which the body is attached.
    * the placement of the body with respect to the joint frame.
    This class is only used in the list Robot.visuals (see below).
    '''
    def __init__(self, name, jointParent, placement):
        self.name = name                  # Name in gepetto viewer
        self.jointParent = jointParent    # ID (int) of the joint 
        self.placement = placement        # placement of the body wrt joint, i.e. bodyMjoint
    
    def place(self, display, oMjoint):
        oMbody = oMjoint*self.placement
        display.place(self.name,oMbody,False)

class Pendulum:
    '''
    Define a class Pendulum with nbJoint joints.
    The members of the class are:
    * viewer: a display encapsulating a gepetto viewer client to create 3D objects and place them.
    * model: the kinematic tree of the robot.
    * data: the temporary variables to be used by the kinematic algorithms.
    * visuals: the list of all the 'visual' 3D objects to render the robot, each element of the list being
    an object Visual (see above).    
    '''

    def __init__(self, n_actions, arms_length, weights, nbJoint=1, noise_stddev=0.0):
        '''Create a Pinocchio model of a N-pendulum, with N the argument <nbJoint>.'''
        assert len(arms_length)==nbJoint
        
        self.viewer         = Display()
        self.visuals        = []
        self.model          = pin.Model()
        self.nbJoints       = nbJoint
        self.length         = arms_length
        self.mass           = self.length
        self.createPendulum()
        self.data           = self.model.createData()
        self.noise_stddev   = noise_stddev
        self.weights        = weights
        
        self.tot_length = 0.0
        for i in range(len(self.length)):
            self.tot_length += self.length[i]

        self.q0         = np.zeros(self.model.nq)

        self.DT         = 5e-2                  # Time step length
        self.NDT        = 1                     # Number of Euler steps per integration (internal)
        self.Kf         = .10                   # Friction coefficient
        self.vmax       = 8.0                   # Max velocity (clipped if larger)
        self.umax       = 2.0*(5**(nbJoint-1))   # Max torque   (clipped if larger)
        self.withSinCos = False                 # If true, state is [cos(q),sin(q),qdot], else [q,qdot]
        
        self.set_Tolerance(1e-3)
        
        self.dsu            = n_actions                 # Number of discretization steps for joint torque
        self.DU             = 2*self.umax/self.dsu      # discretization resolution for joint torque
        self.action_space   = np.linspace(-self.umax, self.umax, self.dsu, dtype=np.float32)
        self.n_combos       = np.power(self.dsu, self.nbJoints)

    def createPendulum(self, rootId=0, prefix='', jointPlacement=None):
        color   = [red,green,blue,transparency] = [0.0,0.0,0.0,1.0]
        color_joint = [0.5,0.5,0.5,1.0]

        jointId = rootId
        jointPlacement     = jointPlacement if jointPlacement!=None else pin.SE3.Identity()
        for i in range(self.nbJoints):
            inertia = pin.Inertia(self.mass[i],
                                       np.array([0.0,0.0,self.length[i]/2]).T,
                                       self.mass[i]/5*np.diagflat([ 1e-2,self.length[i]**2,  1e-2 ]) )
            istr = str(i)
            name      = prefix+"joint"+istr
            jointName = name+"_joint"
            jointId   = self.model.addJoint(jointId, pin.JointModelRY(), jointPlacement, jointName)
            self.model.appendBodyToJoint(jointId, inertia, pin.SE3.Identity())
            try:self.viewer.viewer.gui.addSphere('world/'+prefix+'sphere'+istr, 0.15, color_joint)
            except: pass
            self.visuals.append( Visual('world/'+prefix+'sphere'+istr, jointId, pin.SE3.Identity()) )
            try:self.viewer.viewer.gui.addCapsule('world/'+prefix+'arm'+istr, .1, .8*self.length[i], color)
            except:pass
            self.visuals.append( Visual('world/'+prefix+'arm'+istr, jointId,
                                        pin.SE3(np.eye(3), np.array([0.,0.,self.length[i]/2]))))
            jointPlacement     = pin.SE3(np.eye(3), np.array([0.0,0.0,self.length[i]]).T)

        self.model.addFrame( pin.Frame('tip', jointId, 0, jointPlacement, pin.FrameType.OP_FRAME) )

    def display(self, q):
        ''' Display the robot in the viewer '''
        pin.forwardKinematics(self.model, self.data,q)
        for visual in self.visuals:
            visual.place( self.viewer, self.data.oMi[visual.jointParent] )
        self.viewer.viewer.gui.refresh()


    ''' Size of the q vector '''
    @property 
    def nq(self): return self.model.nq 
    ''' Size of the v vector '''
    @property
    def nv(self): return self.model.nv
    ''' Size of the x vector '''
    @property
    def nx(self): return self.nq+self.nv
#    @property
#    def nobs(self): return self.nx+self.withSinCos
    ''' Size of the u vector '''
    @property
    def nu(self): return self.nv

    def reset(self, x0=None):
        ''' Reset the state of the environment to x0 '''
        if x0 is None: 
            q0 = np.pi*(np.random.rand(self.nq)*2-1)
            v0 = np.random.rand(self.nv)*2-1
            x0 = np.vstack([q0,v0])
        assert len(np.reshape(x0, (self.nx,1)))==self.nx
        self.x = x0.copy()
        self.r = 0.0
        return self.obs(self.x)
    
    def set_Tolerance(self, tolerance):
        self.tolerance = tolerance
        # The layout of the goal vector is [q, q_dot]
        self.up_goal            = [self.tolerance, self.tolerance]
        self.goal               = [[0.], [0.]]
        self.down_goal          = [-self.tolerance, -self.tolerance]
        # The layout of the goal vector is [X_ee, Y_ee, q_dot1, q_dot2]
        self.up_goal_2dofs      = [self.tolerance, self.tot_length+self.tolerance, self.tolerance, self.tolerance]
        self.goal_2dofs         = [0., self.tot_length, 0., 0.]
        self.down_goal_2dofs    = [-self.tolerance, self.tot_length-self.tolerance, -self.tolerance, -self.tolerance]
    
    def is_goal_reached(self, state):
        ''' Check if the position is within a neighbourhood of the goal position
            with velocity within a neighbourhood of the goal velocity
        '''
        state[0] = self.to_AbsRefSys(state[0])
        state = np.reshape(state, (self.nx,1))
        reached = [False]*len(state)
        
        # Single DOF case
        if self.nbJoints==1:
            for i in range(len(state)):
                if state[i]>=self.down_goal[i]:
                    if state[i]<=self.up_goal[i]:
                        reached[i] = True
                    else:
                        reached[i] = False
                else:
                    reached[i] = False
        # Multiple DOFs case
        elif self.nbJoints>=2:
            L = self.length
            X_ee = 0.0
            Y_ee = 0.0
            x_2dofs = np.zeros(self.nbJoints+2)
            # Calculate End-Effector position
            for i in range(self.nbJoints):
                X_ee += L[i]*np.sin(state[i])
                Y_ee += L[i]*np.cos(state[i])
                x_2dofs[i+2] = state[i+self.nbJoints]
            x_2dofs[0] = X_ee
            x_2dofs[1] = Y_ee
            for i in range(len(state)):
                if x_2dofs[i]>=self.down_goal_2dofs[i]:
                    if x_2dofs[i]<=self.up_goal_2dofs[i]:
                        reached[i] = True
                    else:
                        reached[i] = False
                else:
                    reached[i] = False
        
        done = True
        for i in range(len(reached)):
            done = done and reached[i]
        
        return done
    
    def to_AbsRefSys(self, state):
        ''' Returns the passed state with the absolute angular position
            of the joints.
        '''
        n_theta = self.nbJoints
        for i in range(n_theta-1):
            state[i+1] += state[i]
            if state[i]<np.pi and state[i+1]>np.pi:
                state[i+1] = -(2*np.pi-abs(state[i+1]))
            elif state[i]>(-np.pi) and state[i+1]<(-np.pi):
                state[i+1] = 2*np.pi-abs(state[i+1])
        
        return state

    def step(self, iu):
        ''' Simulate one time step '''
        assert(len(iu)==self.nu)
        
        cost = self.compute_cost(self.obs(self.x), iu)
        
        if len(iu)>=2:
            iu = np.transpose(iu.numpy())
        
        _, self.r = self.dynamics(np.reshape(self.x, self.nx), iu)
        
        done = self.is_goal_reached(self.obs(self.x))

        return self.obs(self.x), cost, done

    def obs(self, x):
        ''' Compute the observation of the state '''
        if self.withSinCos:
            return np.vstack([ np.vstack([np.cos(qi),np.sin(qi)]) for qi in x[:self.nq] ] 
                             + [x[self.nq:]],)
        else: return x.copy()

    def tip(self, q):
        '''Return the altitude of pendulum tip'''
        pin.framesKinematics(self.model, self.data,q)
        return self.data.oMf[1].translation[2,0]

    def compute_cost(self, x, u):
        ''' Copmutes the cost based on the current state
            and the applied control
        '''
        Q_WEIGHT = np.full(np.size(x,1), self.weights[0])
        V_WEIGHT = np.full(np.size(x,1), self.weights[1])
        X_WEIGHT = np.diag(np.append(Q_WEIGHT, V_WEIGHT))
        U_WEIGHT = np.diag(np.full(len(u), self.weights[2]))
        
        x[0] = self.to_AbsRefSys(x[0])
        
        # Single DOF case
        if self.nbJoints==1:
            delta_x = self.goal - x
        # Multiple DOFs case
        elif self.nbJoints>=2:
            x = np.reshape(x, (self.nx,1))
            L = self.length
            X_ee = 0.0
            Y_ee = 0.0
            x_2dofs = np.zeros(self.nbJoints+2)
            # Calculate End-Effector position
            for i in range(self.nbJoints):
                X_ee += L[i]*np.sin(x[i])
                Y_ee += L[i]*np.cos(x[i])
                x_2dofs[i+2] = x[i+self.nbJoints]
            x_2dofs[0] = X_ee
            x_2dofs[1] = Y_ee
            
            delta_x = self.goal_2dofs - x_2dofs
        
        cost = np.transpose(delta_x) @ X_WEIGHT @ delta_x
        if len(u)==1:
            #If u is a scalar
            cost += u*U_WEIGHT*u
        elif len(u)>=2:
            # If u is a vector
            cost += np.transpose(u) @ U_WEIGHT @ u
        
        return cost

    def dynamics(self, x, u, display=False):
        '''
        Dynamic function: x,u -> xnext=f(x,y).
        Put the result in x (the initial value is destroyed). 
        Also compute the cost of taking this step.
        Return x for convenience along with the cost.
        '''

        modulePi = lambda th: (th+np.pi)%(2*np.pi)-np.pi
        sumsq    = lambda x : np.sum(np.square(x))

        cost = 0.0
        q = modulePi(x[:self.nq])
        v = x[self.nq:]
        
        u = np.clip(np.reshape(np.array(u),self.nu),-self.umax,self.umax)

        DT = self.DT/self.NDT
        for i in range(self.NDT):
            pin.computeAllTerms(self.model,self.data,q,v)
            M   = self.data.M
            b   = self.data.nle
            a   = (inv(M)*(u-self.Kf*v-b) if self.nbJoints==1
                   else np.dot(inv(M), (u-self.Kf*v-b)))
            a   = a.reshape(self.nv) + np.random.randn(self.nv)*self.noise_stddev
            self.a = a

            q    += (v+0.5*DT*a)*DT
            v    += a*DT
            cost += (sumsq(q) + 1e-1*sumsq(v) + 1e-3*sumsq(u))*DT # cost function

            if display:
                self.display(q)
                time.sleep(1e-4)

        x[:self.nq] = modulePi(q)
        x[self.nq:] = np.clip(v,-self.vmax,self.vmax)
        
        return x, -cost
     
    def render(self):
        if self.nbJoints==1:
            q = self.x[:self.nq]
        else:
            x = np.reshape(self.x,-1) 
            q = x[:self.nq]
        self.display(q)
        time.sleep(self.DT/6.5)
