#include <Python.h>
#include <structmember.h>

const float pi = 3.1415926535897932f;

struct vec_t {
    float x, y, z;
};

struct quat_t {
    float x, y, z, w;
};

struct trans_t {
    vec_t position;
    quat_t rotation;
    float scale;
};

struct vert_t {
    vec_t vertex;
    vec_t normal;
    vec_t color;
};

static inline vec_t normalize(const vec_t & v) {
    const float l = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return {v.x * l, v.y * l, v.z * l};
}

static const trans_t identity = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}, 1.0f};

static inline vec_t transform_vertex(const trans_t & t, const vec_t & v) {
    const float tx = v.y * t.rotation.z - t.rotation.y * v.z - t.rotation.w * v.x;
    const float ty = t.rotation.x * v.z - v.x * t.rotation.z - t.rotation.w * v.y;
    const float tz = v.x * t.rotation.y - t.rotation.x * v.y - t.rotation.w * v.z;
    return {
      t.position.x + (v.x + (ty * t.rotation.z - t.rotation.y * tz) * 2.0f) * t.scale,
      t.position.y + (v.y + (t.rotation.x * tz - tx * t.rotation.z) * 2.0f) * t.scale,
      t.position.z + (v.z + (tx * t.rotation.y - t.rotation.x * ty) * 2.0f) * t.scale,
    };
}

static inline vec_t transform_normal(const trans_t & t, const vec_t & n) {
    const float tx = n.y * t.rotation.z - t.rotation.y * n.z - t.rotation.w * n.x;
    const float ty = t.rotation.x * n.z - n.x * t.rotation.z - t.rotation.w * n.y;
    const float tz = n.x * t.rotation.y - t.rotation.x * n.y - t.rotation.w * n.z;
    return {
      n.x + (ty * t.rotation.z - t.rotation.y * tz) * 2.0f,
      n.y + (t.rotation.x * tz - tx * t.rotation.z) * 2.0f,
      n.z + (tx * t.rotation.y - t.rotation.x * ty) * 2.0f,
    };
}

static inline quat_t quatmul(const quat_t & a, const quat_t & b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

static inline trans_t apply_transform(const trans_t & a, const trans_t & b) {
    return {
        transform_vertex(a, b.position),
        quatmul(a.rotation, b.rotation),
        a.scale * b.scale,
    };
}

static inline vert_t apply_transform(const trans_t & a, const vert_t & b) {
    return {
        transform_vertex(a, b.vertex),
        transform_normal(a, b.normal),
        b.color,
    };
}

struct Mesh {
    PyObject_HEAD
    Mesh * parent;
    Mesh * slibling;
    Mesh * child;
    trans_t local_transform;
    trans_t world_transform;
    int vertex_count;
    vert_t * vertex;
};

struct Scene {
    PyObject_HEAD
    Mesh * base;
};

static PyTypeObject * Mesh_type;
static PyTypeObject * Scene_type;
static PyObject * default_random_uniform;

static Mesh * meth_empty(PyObject * self, PyObject * args, PyObject * kwargs) {
    Mesh * res = PyObject_New(Mesh, Mesh_type);
    res->parent = NULL;
    res->slibling = NULL;
    res->child = NULL;
    res->local_transform = identity;
    res->world_transform = identity;
    res->vertex_count = 0;
    res->vertex = NULL;
    return res;
}

static Mesh * meth_plane(PyObject * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"width", "length", "color", NULL};

    float width, length;
    vec_t color = {1.0f, 1.0f, 1.0f};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ff|(fff)", (char **)keywords, &width, &length, &color.x, &color.y, &color.z)) {
        return NULL;
    }

    const float sx = width * 0.5f;
    const float sy = length * 0.5f;

    Mesh * res = PyObject_New(Mesh, Mesh_type);
    res->parent = NULL;
    res->slibling = NULL;
    res->child = NULL;
    res->local_transform = identity;
    res->world_transform = identity;
    res->vertex_count = 6;
    res->vertex = (vert_t *)PyMem_Malloc(res->vertex_count * sizeof(vert_t));
    res->vertex[0] = {{-sx, -sy, 0.0f}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[1] = {{sx, -sy, 0.0f}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[2] = {{sx, sy, 0.0f}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[3] = {{sx, sy, 0.0f}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[4] = {{-sx, sy, 0.0f}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[5] = {{-sx, -sy, 0.0f}, {0.0f, 0.0f, 1.0f}, color};
    return res;
}

static Mesh * meth_box(PyObject * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"width", "length", "height", "color", NULL};

    float width, length, height;
    vec_t color = {1.0f, 1.0f, 1.0f};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fff|(fff)", (char **)keywords, &width, &length, &height, &color.x, &color.y, &color.z)) {
        return NULL;
    }

    const float sx = width * 0.5f;
    const float sy = length * 0.5f;
    const float sz = height * 0.5f;

    Mesh * res = PyObject_New(Mesh, Mesh_type);
    res->parent = NULL;
    res->slibling = NULL;
    res->child = NULL;
    res->local_transform = identity;
    res->world_transform = identity;
    res->vertex_count = 36;
    res->vertex = (vert_t *)PyMem_Malloc(res->vertex_count * sizeof(vert_t));
    res->vertex[0] = {{-sx, -sy, -sz}, {0.0f, 0.0f, -1.0f}, color};
    res->vertex[1] = {{-sx, sy, -sz}, {0.0f, 0.0f, -1.0f}, color};
    res->vertex[2] = {{sx, sy, -sz}, {0.0f, 0.0f, -1.0f}, color};
    res->vertex[3] = {{sx, sy, -sz}, {0.0f, 0.0f, -1.0f}, color};
    res->vertex[4] = {{sx, -sy, -sz}, {0.0f, 0.0f, -1.0f}, color};
    res->vertex[5] = {{-sx, -sy, -sz}, {0.0f, 0.0f, -1.0f}, color};
    res->vertex[6] = {{-sx, -sy, sz}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[7] = {{sx, -sy, sz}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[8] = {{sx, sy, sz}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[9] = {{sx, sy, sz}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[10] = {{-sx, sy, sz}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[11] = {{-sx, -sy, sz}, {0.0f, 0.0f, 1.0f}, color};
    res->vertex[12] = {{-sx, -sy, -sz}, {0.0f, -1.0f, 0.0f}, color};
    res->vertex[13] = {{sx, -sy, -sz}, {0.0f, -1.0f, 0.0f}, color};
    res->vertex[14] = {{sx, -sy, sz}, {0.0f, -1.0f, 0.0f}, color};
    res->vertex[15] = {{sx, -sy, sz}, {0.0f, -1.0f, 0.0f}, color};
    res->vertex[16] = {{-sx, -sy, sz}, {0.0f, -1.0f, 0.0f}, color};
    res->vertex[17] = {{-sx, -sy, -sz}, {0.0f, -1.0f, 0.0f}, color};
    res->vertex[18] = {{sx, -sy, -sz}, {1.0f, 0.0f, 0.0f}, color};
    res->vertex[19] = {{sx, sy, -sz}, {1.0f, 0.0f, 0.0f}, color};
    res->vertex[20] = {{sx, sy, sz}, {1.0f, 0.0f, 0.0f}, color};
    res->vertex[21] = {{sx, sy, sz}, {1.0f, 0.0f, 0.0f}, color};
    res->vertex[22] = {{sx, -sy, sz}, {1.0f, 0.0f, 0.0f}, color};
    res->vertex[23] = {{sx, -sy, -sz}, {1.0f, 0.0f, 0.0f}, color};
    res->vertex[24] = {{sx, sy, -sz}, {0.0f, 1.0f, 0.0f}, color};
    res->vertex[25] = {{-sx, sy, -sz}, {0.0f, 1.0f, 0.0f}, color};
    res->vertex[26] = {{-sx, sy, sz}, {0.0f, 1.0f, 0.0f}, color};
    res->vertex[27] = {{-sx, sy, sz}, {0.0f, 1.0f, 0.0f}, color};
    res->vertex[28] = {{sx, sy, sz}, {0.0f, 1.0f, 0.0f}, color};
    res->vertex[29] = {{sx, sy, -sz}, {0.0f, 1.0f, 0.0f}, color};
    res->vertex[30] = {{-sx, sy, -sz}, {-1.0f, 0.0f, 0.0f}, color};
    res->vertex[31] = {{-sx, -sy, -sz}, {-1.0f, 0.0f, 0.0f}, color};
    res->vertex[32] = {{-sx, -sy, sz}, {-1.0f, 0.0f, 0.0f}, color};
    res->vertex[33] = {{-sx, -sy, sz}, {-1.0f, 0.0f, 0.0f}, color};
    res->vertex[34] = {{-sx, sy, sz}, {-1.0f, 0.0f, 0.0f}, color};
    res->vertex[35] = {{-sx, sy, -sz}, {-1.0f, 0.0f, 0.0f}, color};
    return res;
}

static Mesh * meth_cylinder(PyObject * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"radius", "height", "resolution", "color", NULL};

    float radius, height;
    int resolution = 16;
    vec_t color = {1.0f, 1.0f, 1.0f};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ff|i(fff)", (char **)keywords, &radius, &height, &resolution, &color.x, &color.y, &color.z)) {
        return NULL;
    }

    Mesh * res = PyObject_New(Mesh, Mesh_type);
    res->parent = NULL;
    res->slibling = NULL;
    res->child = NULL;
    res->local_transform = identity;
    res->world_transform = identity;
    res->vertex_count = resolution * 12;
    res->vertex = (vert_t *)PyMem_Malloc(res->vertex_count * sizeof(vert_t));
    vert_t * ptr = res->vertex;

    const float top = height * 0.5f;
    const float bottom = -height * 0.5f;

    for (int i = 0; i < resolution; ++i) {
        const float a1 = pi * 2.0f * i / resolution;
        const float a2 = pi * 2.0f * (i + 1) / resolution;
        const float c1 = cosf(a1);
        const float s1 = sinf(a1);
        const float c2 = cosf(a2);
        const float s2 = sinf(a2);
        *ptr++ = {{0.0f, 0.0f, bottom}, {0.0f, 0.0f, -1.0f}, color};
        *ptr++ = {{c2 * radius, s2 * radius, bottom}, {0.0f, 0.0f, -1.0f}, color};
        *ptr++ = {{c1 * radius, s1 * radius, bottom}, {0.0f, 0.0f, -1.0f}, color};
        *ptr++ = {{c1 * radius, s1 * radius, bottom}, {c1, s1, 1.0f}, color};
        *ptr++ = {{c2 * radius, s2 * radius, bottom}, {c2, s2, 1.0f}, color};
        *ptr++ = {{c1 * radius, s1 * radius, top}, {c1, s1, 1.0f}, color};
        *ptr++ = {{c1 * radius, s1 * radius, top}, {c1, s1, 1.0f}, color};
        *ptr++ = {{c2 * radius, s2 * radius, bottom}, {c2, s2, 1.0f}, color};
        *ptr++ = {{c2 * radius, s2 * radius, top}, {c2, s2, 1.0f}, color};
        *ptr++ = {{0.0f, 0.0f, top}, {0.0f, 0.0f, 1.0f}, color};
        *ptr++ = {{c1 * radius, s1 * radius, top}, {0.0f, 0.0f, 1.0f}, color};
        *ptr++ = {{c2 * radius, s2 * radius, top}, {0.0f, 0.0f, 1.0f}, color};
    }

    return res;
}

static Mesh * meth_uvsphere(PyObject * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"radius", "resolution", "color", NULL};

    float radius;
    int resolution = 16;
    vec_t color = {1.0f, 1.0f, 1.0f};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f|i(fff)", (char **)keywords, &radius, &resolution, &color.x, &color.y, &color.z)) {
        return NULL;
    }

    resolution = resolution < 8 ? 8 : resolution > 128 ? 128 : resolution;

    int half_resolution = resolution / 2;

    Mesh * res = PyObject_New(Mesh, Mesh_type);
    res->parent = NULL;
    res->slibling = NULL;
    res->child = NULL;
    res->local_transform = identity;
    res->world_transform = identity;
    res->vertex_count = resolution * (half_resolution - 1) * 12;
    res->vertex = (vert_t *)PyMem_Malloc(res->vertex_count * sizeof(vert_t));
    vert_t * ptr = res->vertex;

    for (int i = 0; i < half_resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            const float a1 = pi * 2.0f * j / resolution;
            const float a2 = pi * 2.0f * (j + 1) / resolution;
            const float a3 = pi * i / half_resolution - pi * 0.5f;
            const float a4 = pi * (i + 1) / half_resolution - pi * 0.5f;
            const float c1 = cosf(a1);
            const float s1 = sinf(a1);
            const float c2 = cosf(a2);
            const float s2 = sinf(a2);
            const float c3 = cosf(a3);
            const float s3 = sinf(a3);
            const float c4 = cosf(a4);
            const float s4 = sinf(a4);
            if (i) {
                *ptr++ = {{c1 * c3 * radius, s1 * c3 * radius, s3 * radius}, {c1 * c3, s1 * c3, s3}, color};
                *ptr++ = {{c2 * c3 * radius, s2 * c3 * radius, s3 * radius}, {c2 * c3, s2 * c3, s3}, color};
                *ptr++ = {{c1 * c4 * radius, s1 * c4 * radius, s4 * radius}, {c1 * c4, s1 * c4, s4}, color};
            }
            if (i != half_resolution - 1) {
                *ptr++ = {{c1 * c4 * radius, s1 * c4 * radius, s4 * radius}, {c1 * c4, s1 * c4, s4}, color};
                *ptr++ = {{c2 * c3 * radius, s2 * c3 * radius, s3 * radius}, {c2 * c3, s2 * c3, s3}, color};
                *ptr++ = {{c2 * c4 * radius, s2 * c4 * radius, s4 * radius}, {c2 * c4, s2 * c4, s4}, color};
            }
        }
    }

    return res;
}

static Mesh * meth_icosphere(PyObject * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"radius", "resolution", "color", NULL};

    float radius;
    int resolution = 1;
    vec_t color = {1.0f, 1.0f, 1.0f};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f|i(fff)", (char **)keywords, &radius, &resolution, &color.x, &color.y, &color.z)) {
        return NULL;
    }

    resolution = resolution < 1 ? 1 : resolution > 8 ? 8 : resolution;

    Mesh * res = PyObject_New(Mesh, Mesh_type);
    res->parent = NULL;
    res->slibling = NULL;
    res->child = NULL;
    res->local_transform = identity;
    res->world_transform = identity;
    res->vertex_count = 60 * (1 << ((resolution - 1) * 2));
    res->vertex = (vert_t *)PyMem_Malloc(res->vertex_count * sizeof(vert_t));
    vert_t * ptr = res->vertex + res->vertex_count - 60;

    for (int i = 0; i < 5; ++i) {
        const float a1 = pi * 2.0f * i / 5;
        const float a2 = pi * 2.0f * (i + 1) / 5;
        const float c0 = cosf(atanf(0.5f));
        const float s0 = sinf(atanf(0.5f));
        const float c1 = cosf(a1 - pi * 0.2f);
        const float s1 = sinf(a1 - pi * 0.2f);
        const float c2 = cosf(a2 - pi * 0.2f);
        const float s2 = sinf(a2 - pi * 0.2f);
        const float c3 = cosf(a1);
        const float s3 = sinf(a1);
        const float c4 = cosf(a2);
        const float s4 = sinf(a2);
        (ptr++)->vertex = {0.0f, 0.0f, -1.0f};
        (ptr++)->vertex = {c2 * c0, s2 * c0, -s0};
        (ptr++)->vertex = {c1 * c0, s1 * c0, -s0};
        (ptr++)->vertex = {c1 * c0, s1 * c0, -s0};
        (ptr++)->vertex = {c2 * c0, s2 * c0, -s0};
        (ptr++)->vertex = {c3 * c0, s3 * c0, s0};
        (ptr++)->vertex = {c3 * c0, s3 * c0, s0};
        (ptr++)->vertex = {c2 * c0, s2 * c0, -s0};
        (ptr++)->vertex = {c4 * c0, s4 * c0, s0};
        (ptr++)->vertex = {0.0f, 0.0f, 1.0f};
        (ptr++)->vertex = {c3 * c0, s3 * c0, s0};
        (ptr++)->vertex = {c4 * c0, s4 * c0, s0};
    }

    for (int i = 1; i < resolution; ++i) {
        int triangles = 20 * (1 << ((i - 1) * 2));
        vert_t * src = res->vertex + res->vertex_count - triangles * 3;
        vert_t * dst = res->vertex + res->vertex_count - triangles * 12;
        while (triangles--) {
            vec_t a = (src++)->vertex;
            vec_t b = (src++)->vertex;
            vec_t c = (src++)->vertex;
            vec_t d = normalize({a.x + b.x, a.y + b.y, a.z + b.z});
            vec_t e = normalize({b.x + c.x, b.y + c.y, b.z + c.z});
            vec_t f = normalize({c.x + a.x, c.y + a.y, c.z + a.z});
            (dst++)->vertex = a;
            (dst++)->vertex = d;
            (dst++)->vertex = f;
            (dst++)->vertex = d;
            (dst++)->vertex = b;
            (dst++)->vertex = e;
            (dst++)->vertex = f;
            (dst++)->vertex = d;
            (dst++)->vertex = e;
            (dst++)->vertex = f;
            (dst++)->vertex = e;
            (dst++)->vertex = c;
        }
    }

    for (int i = 0; i < res->vertex_count; ++i) {
        const vec_t & v = res->vertex[i].vertex;
        res->vertex[i] = {{v.x * radius, v.y * radius, v.z * radius}, v, color};
    }

    return res;
}

static Mesh * meth_mesh(PyObject * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"mesh", NULL};

    Py_buffer view = {};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*", (char **)keywords, &view)) {
        return NULL;
    }

    Mesh * res = PyObject_New(Mesh, Mesh_type);
    res->parent = NULL;
    res->slibling = NULL;
    res->child = NULL;
    res->local_transform = identity;
    res->world_transform = identity;
    res->vertex_count = (int)(view.len / sizeof(vert_t));
    res->vertex = (vert_t *)PyMem_Malloc(res->vertex_count * sizeof(vert_t));
    memcpy(res->vertex, view.buf, view.len);

    PyBuffer_Release(&view);
    return res;
}

static inline float random_float(PyObject * uniform) {
    PyObject * res = PyObject_CallFunction(uniform, NULL);
    float x = (float)PyFloat_AsDouble(res);
    Py_DECREF(res);
    return x;
}

static inline quat_t random_rotation(const float u1, const float u2, const float u3) {
    return {
        sqrtf(1.0f - u1) * sinf(2.0f * pi * u2),
        sqrtf(1.0f - u1) * cosf(2.0f * pi * u2),
        sqrtf(u1) * sinf(2.0f * pi * u3),
        sqrtf(u1) * cosf(2.0f * pi * u3),
    };
}

static PyObject * meth_random_rotation(PyObject * self, PyObject ** args, Py_ssize_t nargs) {
    PyObject * uniform = default_random_uniform;
    if (nargs == 1) {
        uniform = args[0];
    }
    const quat_t & q = random_rotation(random_float(uniform), random_float(uniform), random_float(uniform));
    return Py_BuildValue("(ffff)", q.x, q.y, q.z, q.w);
}

static PyObject * meth_random_axis(PyObject * self, PyObject ** args, Py_ssize_t nargs) {
    PyObject * uniform = default_random_uniform;
    if (nargs == 1) {
        uniform = args[0];
    }
    const quat_t & q = random_rotation(random_float(uniform), random_float(uniform), random_float(uniform));
    const float x = (q.x * q.z + q.y * q.w) * 2.0f;
    const float y = (q.y * q.z - q.x * q.w) * 2.0f;
    const float z = 1.0f - (q.x * q.x + q.y * q.y) * 2.0f;
    return Py_BuildValue("(fff)", x, y, z);
}

static PyObject * meth_euler(PyObject * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"x", "y", "z", NULL};

    float x = 0.0f, y = 0.0f, z = 0.0f;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|fff", (char **)keywords, &x, &y, &z)) {
        return NULL;
    }

    const quat_t & qx = {sinf(x * 0.5f), 0.0f, 0.0f, cosf(x * 0.5f)};
    const quat_t & qy = {0.0f, sinf(y * 0.5f), 0.0f, cosf(y * 0.5f)};
    const quat_t & qz = {0.0f, 0.0f, sinf(z * 0.5f), cosf(z * 0.5f)};
    const quat_t & q = quatmul(qx, quatmul(qy, qz));
    return Py_BuildValue("(ffff)", q.x, q.y, q.z, q.w);
}

static Scene * meth_scene(PyObject * self, PyObject * args, PyObject * kwargs) {
    Scene * res = PyObject_New(Scene, Scene_type);
    res->base = PyObject_New(Mesh, Mesh_type);
    res->base->parent = NULL;
    res->base->slibling = NULL;
    res->base->child = NULL;
    res->base->local_transform = identity;
    res->base->world_transform = identity;
    res->base->vertex_count = 0;
    res->base->vertex = NULL;
    return res;
}

static PyObject * Mesh_meth_add(Mesh * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"mesh", NULL};

    Mesh * mesh;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", (char **)keywords, Mesh_type, &mesh)) {
        return NULL;
    }

    Py_INCREF(mesh);
    mesh->parent = self;
    mesh->slibling = self->child;
    self->child = mesh;
    Py_RETURN_NONE;
}

static PyObject * Mesh_meth_paint(Mesh * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"color", NULL};

    vec_t color;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fff", (char **)keywords, &color.x, &color.y, &color.z)) {
        return NULL;
    }

    for (int i = 0; i < self->vertex_count; ++i) {
        self->vertex[i].color = color;
    }
    Py_RETURN_NONE;
}

static PyObject * Scene_meth_add(Scene * self, PyObject * args, PyObject * kwargs) {
    return Mesh_meth_add(self->base, args, kwargs);
}

static PyObject * Scene_meth_bake(Scene * self, PyObject * args) {
    Mesh * stack[1024];
    int stack_index;

    stack_index = 0;
    stack[0] = self->base->child;
    int total_vertex_count = 0;
    while (true) {
        Mesh * mesh = stack[stack_index];
        if (mesh) {
            mesh->world_transform = apply_transform(mesh->parent->world_transform, mesh->local_transform);
            total_vertex_count += mesh->vertex_count;
            stack[stack_index] = mesh->slibling;
            if (mesh->child) {
                stack[++stack_index] = mesh->child;
            }
        } else {
            --stack_index;
            if (stack_index < 0) {
                break;
            }
        }
    }

    PyObject * res = PyBytes_FromStringAndSize(NULL, total_vertex_count * sizeof(vert_t));
    vert_t * ptr = (vert_t *)PyBytes_AsString(res);

    stack_index = 0;
    stack[0] = self->base->child;
    while (true) {
        Mesh * mesh = stack[stack_index];
        if (mesh) {
            const trans_t & t = mesh->world_transform;
            vert_t * src = mesh->vertex;
            int count = mesh->vertex_count;
            while (count--) {
                *ptr++ = apply_transform(t, *src++);
            }
            stack[stack_index] = mesh->slibling;
            if (mesh->child) {
                stack[++stack_index] = mesh->child;
            }
        } else {
            --stack_index;
            if (stack_index < 0) {
                break;
            }
        }
    }

    return res;
}

PyObject * Mesh_get_position(Mesh * self, void * closure) {
    const vec_t & p = self->local_transform.position;
    return Py_BuildValue("(fff)", p.x, p.y, p.z);
}

int Mesh_set_position(Mesh * self, PyObject * value, void * closure) {
    PyObject * tup = PySequence_Tuple(value);
    self->local_transform.position = {
        (float)PyFloat_AsDouble(PyTuple_GET_ITEM(tup, 0)),
        (float)PyFloat_AsDouble(PyTuple_GET_ITEM(tup, 1)),
        (float)PyFloat_AsDouble(PyTuple_GET_ITEM(tup, 2)),
    };
    Py_DECREF(tup);
    return 0;
}

PyObject * Mesh_get_rotation(Mesh * self, void * closure) {
    const quat_t & q = self->local_transform.rotation;
    return Py_BuildValue("(ffff)", q.x, q.y, q.z, q.w);
}

int Mesh_set_rotation(Mesh * self, PyObject * value, void * closure) {
    PyObject * tup = PySequence_Tuple(value);
    self->local_transform.rotation = {
        (float)PyFloat_AsDouble(PyTuple_GET_ITEM(tup, 0)),
        (float)PyFloat_AsDouble(PyTuple_GET_ITEM(tup, 1)),
        (float)PyFloat_AsDouble(PyTuple_GET_ITEM(tup, 2)),
        (float)PyFloat_AsDouble(PyTuple_GET_ITEM(tup, 3)),
    };
    Py_DECREF(tup);
    return 0;
}

PyObject * Mesh_get_scale(Mesh * self, void * closure) {
    return PyFloat_FromDouble(self->local_transform.scale);
}

int Mesh_set_scale(Mesh * self, PyObject * value, void * closure) {
    self->local_transform.scale = (float)PyFloat_AsDouble(value);
    return 0;
}

PyObject * Mesh_get_world_transform(Mesh * self, void * closure) {
    trans_t t = self->local_transform;
    Mesh * ptr = self;
    while (ptr->parent) {
        t = apply_transform(ptr->parent->local_transform, t);
        ptr = ptr->parent;
    }
    const vec_t & p = t.position;
    const quat_t & r = t.rotation;
    return Py_BuildValue("((fff)(ffff)f)", p.x, p.y, p.z, r.x, r.y, r.z, r.w, t.scale);
}

PyObject * Mesh_get_mem(Mesh * self, void * closure) {
    return PyMemoryView_FromMemory((char *)self->vertex, self->vertex_count * sizeof(vert_t), PyBUF_WRITE);
}

static void default_dealloc(PyObject * self) {
    Py_TYPE(self)->tp_free(self);
}

static PyMethodDef Mesh_methods[] = {
    {"add", (PyCFunction)Mesh_meth_add, METH_VARARGS | METH_KEYWORDS},
    {"paint", (PyCFunction)Mesh_meth_paint, METH_VARARGS | METH_KEYWORDS},
    {},
};

static PyGetSetDef Mesh_getset[] = {
    {"position", (getter)Mesh_get_position, (setter)Mesh_set_position},
    {"rotation", (getter)Mesh_get_rotation, (setter)Mesh_set_rotation},
    {"scale", (getter)Mesh_get_scale, (setter)Mesh_set_scale},
    {"world_transform", (getter)Mesh_get_world_transform, NULL},
    {"mem", (getter)Mesh_get_mem, NULL},
    {},
};

static PyType_Slot Mesh_slots[] = {
    {Py_tp_methods, Mesh_methods},
    {Py_tp_getset, Mesh_getset},
    {Py_tp_dealloc, default_dealloc},
    {},
};

static PyMethodDef Scene_methods[] = {
    {"add", (PyCFunction)Scene_meth_add, METH_VARARGS | METH_KEYWORDS},
    {"bake", (PyCFunction)Scene_meth_bake, METH_VARARGS | METH_KEYWORDS},
    {},
};

static PyType_Slot Scene_slots[] = {
    {Py_tp_methods, Scene_methods},
    {Py_tp_dealloc, default_dealloc},
    {},
};

static PyType_Spec Mesh_spec = {"meshes.Mesh", sizeof(Mesh), 0, Py_TPFLAGS_DEFAULT, Mesh_slots};
static PyType_Spec Scene_spec = {"meshes.Scene", sizeof(Scene), 0, Py_TPFLAGS_DEFAULT, Scene_slots};

static PyMethodDef module_methods[] = {
    {"empty", (PyCFunction)meth_empty, METH_VARARGS | METH_KEYWORDS},
    {"plane", (PyCFunction)meth_plane, METH_VARARGS | METH_KEYWORDS},
    {"box", (PyCFunction)meth_box, METH_VARARGS | METH_KEYWORDS},
    {"cylinder", (PyCFunction)meth_cylinder, METH_VARARGS | METH_KEYWORDS},
    {"uvsphere", (PyCFunction)meth_uvsphere, METH_VARARGS | METH_KEYWORDS},
    {"icosphere", (PyCFunction)meth_icosphere, METH_VARARGS | METH_KEYWORDS},
    {"mesh", (PyCFunction)meth_mesh, METH_VARARGS | METH_KEYWORDS},
    {"scene", (PyCFunction)meth_scene, METH_VARARGS | METH_KEYWORDS},
    {"random_rotation", (PyCFunction)meth_random_rotation, METH_FASTCALL},
    {"random_axis", (PyCFunction)meth_random_axis, METH_FASTCALL},
    {"euler", (PyCFunction)meth_euler, METH_VARARGS | METH_KEYWORDS},
    {},
};

static PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "meshes", NULL, -1, module_methods};

extern "C" PyObject * PyInit_meshes() {
    PyObject * module = PyModule_Create(&module_def);
    Scene_type = (PyTypeObject *)PyType_FromSpec(&Scene_spec);
    Mesh_type = (PyTypeObject *)PyType_FromSpec(&Mesh_spec);
    PyModule_AddObject(module, "Scene", (PyObject *)Scene_type);
    PyModule_AddObject(module, "Mesh", (PyObject *)Mesh_type);

    PyObject * random = PyImport_ImportModule("random");
    default_random_uniform = PyObject_GetAttrString(random, "random");
    return module;
}
