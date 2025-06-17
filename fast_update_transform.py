from ut_cpp import pl_qcqp

def old_update_transform(x_in, pt, q1s, q2s, norms):
    return pl_qcqp.update_transform(x_in, pt, q1s, q2s, norms)

def update_transform(pt, q1s, q2s, max_iter=10):
    return pl_qcqp.update_transform(pt, q1s, q2s, max_iter)