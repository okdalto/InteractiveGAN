from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, QtWidgets, QtOpenGL
import numpy as np
import pyrr
from pyrr import Vector3, vector, vector3, matrix44

import OpenGL.GL.shaders as shaders
from PIL import Image as Image

import math




# class Ui_MainWindow(QtWidgets.QWidget):
#     def __init__(self, parent=None):
#         super(Ui_MainWindow, self).__init__()
#         self.widget = glWidget()
#         self.button = QtWidgets.QPushButton('Test', self)
#         mainLayout = QtWidgets.QHBoxLayout()

#         mainLayout.addWidget(self.widget)
#         mainLayout.addWidget(self.button)
#         self.setLayout(mainLayout)

class glWidget(QGLWidget):
    def __init__(self, parent=None):
        QGLWidget.__init__(self, parent)
        self.screen_width = 512
        self.screen_height = 512
        self.setMinimumSize(self.screen_width, self.screen_height)
        self.time = 1
        self.rot_angles = {'x':0, 'y':0, 'z':0}

    def load_texture(self, filename):
        texture = 0
        pBitmap = Image.open(filename)
        pBitmapData = np.array(list(pBitmap.getdata()), np.int8)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, pBitmap.size[0], pBitmap.size[1], 0,
            GL_RGB, GL_UNSIGNED_BYTE, pBitmapData
            )
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def make_texture(self):
        texture = 0
        pBitmapData = np.zeros((128,128,3))
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 128, 128, 0,
            GL_RGB, GL_UNSIGNED_BYTE, pBitmapData
            )
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture



    def paintGL(self):
        self.time += 0.1
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(self.rot_angles['x'])
        rot_y = pyrr.Matrix44.from_y_rotation(self.rot_angles['y'])
        rot_z = pyrr.Matrix44.from_z_rotation(self.rot_angles['z'])
        model = rot_x * rot_y * rot_z
        modelView = pyrr.matrix44.multiply(self.view, model)
        normalMatrix = pyrr.matrix44.inverse(modelView).T
        glPatchParameteri(GL_PATCH_VERTICES, 4)

        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniform1f(self.time_loc, self.time)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        # Draw Cube
        glDrawArrays(GL_PATCHES, 0, 4)
        # glDrawArrays(GL_QUADS, 0, 4)
        # glDrawElements(GL_PATCHES, 36, GL_UNSIGNED_INT, None)
        # glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT,  None)
        glFlush()

    def set_model(self, angle, axis=''):
        self.rot_angles[axis] = math.radians(angle)

    def set_texture(self, input_texture):
        # print(input_texture.shape)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 128, 128, 0,
            GL_RGB, GL_UNSIGNED_BYTE, input_texture
            )


    def initializeGL(self):
        # cube = [
        #     -0.5, -0.5,  0.5,  0.0,  0.0,  1.0, 0.0, 0.0,
        #      0.5, -0.5,  0.5,  0.0,  0.0,  1.0, 1.0, 0.0,
        #      0.5,  0.5,  0.5,  0.0,  0.0,  1.0, 1.0, 1.0,
        #     -0.5,  0.5,  0.5,  0.0,  0.0,  1.0, 0.0, 1.0,
 
        #     -0.5, -0.5, -0.5,  1.0,  0.0,  0.0, 0.0, 0.0,
        #      0.5, -0.5, -0.5,  1.0,  0.0,  0.0, 1.0, 0.0,
        #      0.5,  0.5, -0.5,  1.0,  0.0,  0.0, 1.0, 1.0,
        #     -0.5,  0.5, -0.5,  1.0,  0.0,  0.0, 0.0, 1.0,
 
        #      0.5, -0.5, -0.5,  0.0,  1.0,  0.0, 0.0, 0.0,
        #      0.5,  0.5, -0.5,  0.0,  1.0,  0.0, 1.0, 0.0,
        #      0.5,  0.5,  0.5,  0.0,  1.0,  0.0, 1.0, 1.0,
        #      0.5, -0.5,  0.5,  0.0,  1.0,  0.0, 0.0, 1.0,
 
        #     -0.5,  0.5, -0.5, -1.0,  0.0,  0.0, 0.0, 0.0,
        #     -0.5, -0.5, -0.5, -1.0,  0.0,  0.0, 1.0, 0.0,
        #     -0.5, -0.5,  0.5, -1.0,  0.0,  0.0, 1.0, 1.0,
        #     -0.5,  0.5,  0.5, -1.0,  0.0,  0.0, 0.0, 1.0,
 
        #     -0.5, -0.5, -0.5,  0.0, -1.0,  0.0, 0.0, 0.0,
        #      0.5, -0.5, -0.5,  0.0, -1.0,  0.0, 1.0, 0.0,
        #      0.5, -0.5,  0.5,  0.0, -1.0,  0.0, 1.0, 1.0,
        #     -0.5, -0.5,  0.5,  0.0, -1.0,  0.0, 0.0, 1.0,
 
        #      0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 0.0, 0.0,
        #     -0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 1.0, 0.0,
        #     -0.5,  0.5,  0.5,  0.0,  0.0, -1.0, 1.0, 1.0,
        #      0.5,  0.5,  0.5,  0.0,  0.0, -1.0, 0.0, 1.0
        # ]

        cube=[
             -0.7, -0.7, 0, 0, 0, 0, 0, 1,
              0.7, -0.7, 0, 0, 0, 0, 1, 1,
              0.7,  0.7, 0, 0, 0, 0, 1, 0,
             -0.7,  0.7, 0, 0, 0, 0, 0, 0,
        ]
        cube = np.array(cube, dtype=np.float32)

        # indices = [
        #     0, 1, 2,   2, 3, 0,   
        #     4, 5, 6,   6, 7, 4,   
        #     8, 9,10,  10,11, 8,   
        #     12,13,14,  14,15,12,  
        #     16,17,18,  18,19,16,  
        #     20,21,22,  22,23,20   
        # ]
        # indices = np.array(indices, dtype = np.uint32)
        VERTEX_SHADER = """
            #version 410 core

            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            layout (location = 2) in vec2 uv;
            
            uniform mat4 model; 
            uniform mat4 view; 
            uniform mat4 projection; 
            uniform sampler2D ourTexture;

            out VS_OUT {
                vec3 vertColor;
                vec2 uv;
            } vs_out;

            void main() {
                vs_out.vertColor = color;
                vs_out.uv = uv;
                gl_Position = projection * view * model * vec4(position, 1.0f);
            }
        """

        FRAGMENT_SHADER = """
            #version 410 core
            uniform sampler2D ourTexture;
            out vec4 outColor;

            in vec3 normal;
            in vec2 uv;
            in vec3 light_dir; 
            in vec3 camera_dir; 

            void main() {
                vec3 N = normalize(normal);
                vec3 V = normalize(camera_dir);
                vec3 L = normalize(light_dir);
                vec3 H = normalize(N + V);

                float spec = pow(max(0, dot(V, reflect(-L, N))), 16);

                float diff = max(0.0, dot(normalize(light_dir), normalize(normal)));
                vec3 col = texture(ourTexture, uv).xyz;
                //outColor = vec4(normal, 1.0);   
                outColor = vec4(vec3(spec + diff + 0.2) * col, 1.0);   
                //outColor = vec4(L, 1.0);   
            }

        """

        GEOMETRY_SHADER = """
            #version 410 core

            layout (triangles) in;
            layout (triangle_strip, max_vertices = 24) out;

            uniform mat4 model;
            uniform mat4 view; 
            uniform mat4 projection; 
            uniform float time;
            uniform sampler2D ourTexture;
            uniform vec3 camera_pos;

            in vec3 te_color[]; 
            in vec2 te_uv[];

            out vec3 normal;
            out vec2 uv;
            out vec3 light_dir;
            out vec3 camera_dir;

            // Define the 8 corners of a cube (back plane, front plane (counter clockwise))
            vec3 cube_corners[8] = vec3[]  (
                vec3(-1.0,  1.0, -1.0), // left top far
                vec3( 1.0,  1.0, -1.0), // right top far
                vec3(-1.0, -1.0, -1.0), // left bottom far
                vec3( 1.0, -1.0, -1.0), // right bottom far
                vec3( 1.0,  1.0,  1.0), // right top near
                vec3(-1.0,  1.0,  1.0), // left top near
                vec3(-1.0, -1.0,  1.0), // left bottom near
                vec3( 1.0, -1.0,  1.0)  // right bottom near
            );

            #define EMIT_V(POS, UV, NORMAL) \
                light_dir = vec3(cos(time_scaled), sin(time_scaled), sin(time_scaled)) - POS; \
                gl_Position = vec4(POS, 1.0); \
                camera_dir = camera_pos - POS; \
                normal = NORMAL; \
                uv = te_uv[0]; \
                EmitVertex()

            #define EMIT_QUAD(P1, P2, P3, P4, NORMAL) \
                EMIT_V(corners[P1], vec2(0.0, 0.0), NORMAL); \
                EMIT_V(corners[P2], vec2(1.0, 0.0), NORMAL); \
                EMIT_V(corners[P3], vec2(0.0, 1.0), NORMAL); \
                EMIT_V(corners[P4], vec2(1.0, 1.0), NORMAL); \
                EndPrimitive()

            #define CALC_NORMAL(p1, p2, p3, p4) \
                normalize(cross(corners[p1] - corners[p2], corners[p3] - corners[p2]))

            vec3 rotX(vec3 z, float s, float c) {
                vec3 newZ = z;
                newZ.yz = vec2(c*z.y + s*z.z, c*z.z - s*z.y);
                return newZ;
            }

            vec3 rotY(vec3 z, float s, float c) {
                vec3 newZ = z;
                newZ.xz = vec2(c*z.x - s*z.z, c*z.z + s*z.x);
                return newZ;
            }

            vec3 rotZ(vec3 z, float s, float c) {
                vec3 newZ = z;
                newZ.xy = vec2(c*z.x + s*z.y, c*z.y - s*z.x);
                return newZ;
            }

            vec3 rotX(vec3 z, float a) {
                return rotX(z, sin(a), cos(a));
            }
            vec3 rotY(vec3 z, float a) {
                return rotY(z, sin(a), cos(a));
            }

            vec3 rotZ(vec3 z, float a) {
                return rotZ(z, sin(a), cos(a));
            }

            mat4 rotX_mat(float a){
                mat4 m;
                m[0] = vec4(1,      0,       0, 0);
                m[1] = vec4(0, cos(a), -sin(a), 0);
                m[2] = vec4(0, sin(a),  cos(a), 0);
                m[3] = vec4(0,      0,      0,  1);
                return m;
            }

            mat4 rotY_mat(float a){
                mat4 m;
                m[0] = vec4( cos(a), 0,  sin(a), 0);
                m[1] = vec4(      0, 1,       0, 0);
                m[2] = vec4(-sin(a), 0,  cos(a), 0);
                m[3] = vec4(      0, 0,      0,  1);
                return m;
            }

            mat4 rotZ_mat(float a){
                mat4 m;
                m[0] = vec4(cos(a), -sin(a), 0, 0);
                m[1] = vec4(sin(a),  cos(a), 0, 0);
                m[2] = vec4(     0,       0, 1, 0);
                m[3] = vec4(     0,       0, 0, 1);
                return m;
            }

            void main()
            {
                // Calculate the 8 cube corners
                vec3 point = gl_in[0].gl_Position.xyz;
                
                vec3 corners[8];
                int i;
                vec3 tex_value = texture(ourTexture, te_uv[0].xy).xyz;
                vec3 rot_angle = ((tex_value - 0.5) * 2.0) * 3.14159265;
                mat4 rot = rotX_mat(rot_angle.x) * rotY_mat(rot_angle.y) * rotZ_mat(rot_angle.z);  
                for(i = 0; i < 8; i++)
                {
                    vec3 corner = cube_corners[i] * 0.05;
                    corner = (projection * view * model * rot * vec4(corner, 1)).xyz;
                    vec3 pos = point + corner;
                    
                    corners[i] = pos;
                }
                float time_scaled = time * 0.1;
                EMIT_QUAD(3, 2, 0, 1, CALC_NORMAL(3, 2, 0, 1)    ); // back
                EMIT_QUAD(6, 7, 5, 4, CALC_NORMAL(6, 7, 5, 4)    ); // front
                EMIT_QUAD(7, 3, 4, 0, CALC_NORMAL(7, 3, 4, 0)    ); // right
                EMIT_QUAD(2, 6, 1, 5, CALC_NORMAL(2, 6, 1, 5)    ); // left
                EMIT_QUAD(5, 4, 1, 0, CALC_NORMAL(5, 4, 1, 0)    ); // top
                EMIT_QUAD(2, 3, 6, 7, CALC_NORMAL(2, 3, 6, 7)    ); // bottom
                EndPrimitive();
            }        
            """

        TESS_CT_SHADER = """
        #version 410 core

        layout(vertices = 3) out;

        in VS_OUT {
            vec3 vertColor;`
            vec2 uv;
        } tc_in[];

        out vec3 tc_color[];
        out vec2 tc_uv[];

        void main(void)
        {
            gl_TessLevelOuter[0] =32.0;
            gl_TessLevelOuter[1] =32.0;
            gl_TessLevelOuter[2] =32.0;

            gl_TessLevelInner[0] =32.0;

            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
            tc_color[gl_InvocationID] = tc_in[gl_InvocationID].vertColor;
            tc_uv[gl_InvocationID] = tc_in[gl_InvocationID].uv;
        }
        """

        TESS_EV_SHADER = """
        #version 410 core

        layout(triangles, equal_spacing, ccw) in;
        in vec3 tc_color[];
        in vec2 tc_uv[];

        out vec3 te_color;
        out vec2 te_uv;

        void main()
        {	
            gl_Position.xyzw =	gl_in[0].gl_Position.xyzw * gl_TessCoord.x +
                                gl_in[1].gl_Position.xyzw * gl_TessCoord.y +
                                gl_in[2].gl_Position.xyzw * gl_TessCoord.z;
            te_color = tc_color[0] * gl_TessCoord.x + tc_color[1] * gl_TessCoord.y + tc_color[2] * gl_TessCoord.z;
            te_uv = tc_uv[0] * gl_TessCoord.x + tc_uv[1] * gl_TessCoord.y + tc_uv[2] * gl_TessCoord.z;
        }
        """

        TESS_CT_SHADER_QUAD = """
        #version 410 core

        layout(vertices = 4) out;

        in VS_OUT {
            vec3 vertColor;
            vec2 uv;
        } tc_in[];

        out vec3 tc_color[];
        out vec2 tc_uv[];

        void main(void)
        {
            gl_TessLevelOuter[0] =64.0;
            gl_TessLevelOuter[1] =64.0;
            gl_TessLevelOuter[2] =64.0;
            gl_TessLevelOuter[3] =64.0;

            gl_TessLevelInner[0] =64.0;
            gl_TessLevelInner[1] =64.0;

            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
            tc_color[gl_InvocationID] = tc_in[gl_InvocationID].vertColor;
            tc_uv[gl_InvocationID] = tc_in[gl_InvocationID].uv;
        }
        """


        TESS_EV_SHADER_QUAD = """
        #version 410 core

        layout(quads, equal_spacing, ccw) in;
        in vec3 tc_color[];
        in vec2 tc_uv[];

        out vec3 te_color;
        out vec2 te_uv;

        void main()
        {	
            vec3 p0 = mix(gl_in[0].gl_Position.xyz, gl_in[1].gl_Position.xyz, gl_TessCoord.x);
            vec3 p1 = mix(gl_in[3].gl_Position.xyz, gl_in[2].gl_Position.xyz, gl_TessCoord.x);
            vec3 p = mix(p0, p1, gl_TessCoord.y);

            gl_Position.xyzw =	vec4(p, 1);
            te_color = tc_color[0] * gl_TessCoord.x + tc_color[1] * gl_TessCoord.y + tc_color[2] * gl_TessCoord.z;
            vec2 uv0 = mix(tc_uv[0], tc_uv[1], gl_TessCoord.x);
            vec2 uv1 = mix(tc_uv[3], tc_uv[2], gl_TessCoord.x);
            te_uv = mix(uv0, uv1, gl_TessCoord.y);
        }
        """

        vert_shader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        if glGetShaderiv(vert_shader, GL_COMPILE_STATUS) != GL_TRUE:
            info = glGetShaderInfoLog(vert_shader)
            print(info)
        frag_shader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        if glGetShaderiv(frag_shader, GL_COMPILE_STATUS) != GL_TRUE:
            info = glGetShaderInfoLog(frag_shader)
            print(info)
        geometry_shader = shaders.compileShader(GEOMETRY_SHADER, GL_GEOMETRY_SHADER)
        if glGetShaderiv(geometry_shader, GL_COMPILE_STATUS) != GL_TRUE:
            info = glGetShaderInfoLog(geometry_shader)
            print(info)
        tess_control_shader = shaders.compileShader(TESS_CT_SHADER_QUAD, GL_TESS_CONTROL_SHADER)
        if glGetShaderiv(tess_control_shader, GL_COMPILE_STATUS) != GL_TRUE:
            info = glGetShaderInfoLog(tess_control_shader)
            print(info)
        tess_eval_shader = shaders.compileShader(TESS_EV_SHADER_QUAD, GL_TESS_EVALUATION_SHADER)

        # Compile The Program and shaders
        shader = glCreateProgram()
        glAttachShader(shader, frag_shader)
        glAttachShader(shader, vert_shader)
        glAttachShader(shader, geometry_shader)
        glAttachShader(shader, tess_control_shader)
        glAttachShader(shader, tess_eval_shader)
        glLinkProgram(shader)        
        if glGetProgramiv(shader, GL_LINK_STATUS) != GL_TRUE:
            info = glGetShaderInfoLog(shader)
            print(info)

        print(glGetIntegerv(GL_MAX_PATCH_VERTICES))
        
        # Load texture
        self.texture = self.make_texture()


        # Create Buffer object in gpu
        VBO = glGenBuffers(1)
        # Bind the buffer
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, cube.nbytes, cube, GL_STATIC_DRAW)

        #Create EBO
        # EBO = glGenBuffers(1)
        # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glUseProgram(shader)

        # get the position from shader
        position_loc = 0
        glEnableVertexAttribArray(position_loc)
        glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 4*3+4*3+4*2, ctypes.c_void_p(4*0))
        # glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(4*0))

        # # get the color from shader
        color_loc = 1
        glEnableVertexAttribArray(color_loc)
        glVertexAttribPointer(color_loc, 3, GL_FLOAT, GL_FALSE, 4*3+4*3+4*2, ctypes.c_void_p(4*3))
        # glVertexAttribPointer(color_loc, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(4*3))

        uv_loc = 2
        glEnableVertexAttribArray(uv_loc)
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 4*3+4*3+4*2, ctypes.c_void_p(4*3 + 4*3))

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glViewport(0,0,self.screen_width, self.screen_height)
        # glViewport(0,0,100,100)
        self.time_loc = glGetUniformLocation(shader, "time")
        self.model_loc = glGetUniformLocation(shader, "model")
        self.view_loc = glGetUniformLocation(shader, "view")
        self.projection_loc = glGetUniformLocation(shader, "projection")
        self.normal_loc = glGetUniformLocation(shader, "normal_m")
        self.camera_loc = glGetUniformLocation(shader, "camera_pos")

        self.camera_pos = Vector3([0.0, 0.0, 0.1])
        self.camera_front = Vector3([0.0, 0.0, -1.0])
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        self.camera_right = Vector3([1.0, 0.0, 0.0])

        self.view = matrix44.create_look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(90, 1, 0.01, 200)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view)
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, self.projection)
        glUniform3fv(self.camera_loc, 1, self.camera_loc)




# if __name__ == '__main__':    
#     app = QtWidgets.QApplication(sys.argv)    
#     Form = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow(Form)    
#     ui.show()    


#     loop=QtCore.QTimer()
#     loop.timeout.connect(ui.widget.update)
#     loop.start(0)

#     sys.exit(app.exec_())
