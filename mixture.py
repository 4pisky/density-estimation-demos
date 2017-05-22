import numpy as np
from attr import attrib, attrs


def _to_mixture_weights(rel_weights):
    """
    Converts to np.array of floats, normalizes so sum=1.
    """
    weights = np.atleast_1d(np.array(rel_weights, dtype=np.float_, copy=True))
    weights /= weights.sum()
    return weights


def _validate_weights(instance, attribute, value):
    assert value.ndim == 1
    if len(value) != len(instance.models):
        raise ValueError("Weights vector does not match number of models")

def _validate_models(instance, attribute, value):
    model_list = value
    ndim0 = model_list[0].ndim
    for model in model_list:
        if model.ndim != ndim0:
            raise ValueError(
                "Models of different dimensionality supplied")


@attrs
class Mixture(object):
    """
    Represents a mixture model

    Attributes:
    """

    models = attrib()
    weights = attrib(convert=_to_mixture_weights,
                     validator=_validate_weights)

    def joint_pdf(self, x):
        # Could try to zero-allocate, but getting the dimensions
        # right is a little tricky.
        # jpdf = np.zeros(len(x),dtype=np.float_)
        jpdf = None
        for idx, m in enumerate(self.models):
            if jpdf is None:
                jpdf = self.weights[idx] * m.dist.pdf(x)
            else:
                jpdf += self.weights[idx] * m.dist.pdf(x)
        return jpdf


    def joint_sample(self, size, shuffle=True):
        component_sample_sizes = np.random.multinomial(n=size,
                                                       pvals=self.weights)
        component_samples = []
        for idx in range(len(self.models)):
            sample = self.models[idx].dist.rvs(component_sample_sizes[idx])
            component_samples.append(sample)
        mixture_sample = np.concatenate(component_samples)
        if shuffle:
            np.random.shuffle(mixture_sample)
        return mixture_sample



    def _repr_html_(self):
        output=[]
        for idx, mdl in enumerate(self.models):
            output.append(
                """
                <h4>Component {idx}</h4>
                <p>Weight: {weight:0.3f}
                {model}
                </p>
                <div style="clear:left;"></div>
                """.format(idx=idx, weight=self.weights[idx],
                           model = mdl._repr_html_())
            )
        return '\n'.join(output)

    @property
    def ndim(self):
        return self.models[0].ndim
