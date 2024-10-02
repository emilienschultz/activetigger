import { FC, useEffect } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import Select from 'react-select';
import { useGetSimpleModel, useUpdateSimpleModel } from '../core/api';
import { useAppContext } from '../core/context';
import { SimpleModelModel } from '../types';

// TODO: default values + avoid generic parameters

interface SimpleModelManagementProps {
  projectName: string | null;
  currentScheme: string | null;
  availableSimpleModels: Record<string, Record<string, number>>;
  availableFeatures: string[];
}

export const SimpleModelManagement: FC<SimpleModelManagementProps> = ({
  projectName,
  currentScheme,
  availableSimpleModels,
  availableFeatures,
}) => {
  // element from the context
  const {
    appContext: { freqRefreshSimpleModel, currentProject: project },
    setAppContext,
  } = useAppContext();

  // available features
  const features = availableFeatures.map((e) => ({ value: e, label: e }));

  // API call to get the current model (refresh with project)
  const { currentModel, reFetchSimpleModel } = useGetSimpleModel(projectName, currentScheme);
  useEffect(() => {
    reFetchSimpleModel();
  }, [project, reFetchSimpleModel]);

  // function to change refresh frequency
  const refreshFreq = (newValue: number) => {
    setAppContext((prev) => ({ ...prev, freqRefreshSimpleModel: newValue }));
  };

  // create form
  const { register, handleSubmit, control, reset, watch } = useForm<SimpleModelModel>({
    defaultValues: {
      model: currentModel ? currentModel.model : 'liblinear',
      features: Object.values(availableFeatures),
      scheme: currentScheme || undefined,
      params: { cost: 1, C: 32, n_neighbors: 3, alpha: 1, n_estimators: 500, max_features: null },
    },
  });

  // state for the model selected to modify parameters
  const selectedModel = watch('model');

  // hooks to update
  const { updateSimpleModel } = useUpdateSimpleModel(projectName, currentScheme);

  // action when form validated
  const onSubmit: SubmitHandler<SimpleModelModel> = async (formData) => {
    await updateSimpleModel(formData);
    reset();
    // SET THE VALUE FROM THE STATE
  };

  return (
    <div>
      <span className="explanations">Train a prediction model on the current annotated data</span>
      <form onSubmit={handleSubmit(onSubmit)}>
        <label htmlFor="model">Select a model</label>
        <select id="model" {...register('model')}>
          {Object.keys(availableSimpleModels).map((e) => (
            <option key={e}>{e}</option>
          ))}{' '}
        </select>
        {
          //generate_config(selectedSimpleModel)
          (selectedModel == 'liblinear' && (
            <div>
              <label htmlFor="cost">Cost</label>
              <input
                type="number"
                step="1"
                id="cost"
                {...register('params.cost', { valueAsNumber: true })}
              ></input>
            </div>
          )) ||
            (selectedModel == 'knn' && (
              <div>
                <label htmlFor="n_neighbors">Number of neighbors</label>
                <input
                  type="number"
                  step="1"
                  id="n_neighbors"
                  {...register('params.n_neighbors', { valueAsNumber: true })}
                ></input>
              </div>
            )) ||
            (selectedModel == 'lasso' && (
              <div>
                <label htmlFor="c">C</label>
                <input
                  type="number"
                  step="1"
                  id="C"
                  {...register('params.C', { valueAsNumber: true })}
                ></input>
              </div>
            )) ||
            (selectedModel == 'multi_naivebayes' && (
              <div>
                <label htmlFor="alpha">Alpha</label>
                <input
                  type="number"
                  id="alpha"
                  {...register('params.alpha', { valueAsNumber: true })}
                ></input>
                {/* <label htmlFor="fit_prior">
                  Fit prior
                  <input
                    type="checkbox"
                    id="fit_prior"
                    {...register('params.fit_prior')}
                    className="mx-3"
                  />
                </label>
                <label htmlFor="class_prior">Class prior</label>
                <input type="number" id="class_prior" {...register('params.class_prior')}></input> */}
              </div>
            )) ||
            (selectedModel == 'randomforest' && (
              <div>
                <label htmlFor="n_estimators">Number of estimators</label>
                <input
                  type="number"
                  step="1"
                  id="n_estimators"
                  {...register('params.n_estimators', { valueAsNumber: true })}
                ></input>
                <label htmlFor="max_features">Max features</label>
                <input
                  type="number"
                  step="1"
                  id="max_features"
                  {...register('params.max_features', { valueAsNumber: true })}
                ></input>
              </div>
            ))
        }
        <div>
          <label htmlFor="features">Select features</label>
          {/* Specific management of the component with the react-form controller */}
          <Controller
            name="features"
            control={control}
            render={({ field: { value, onChange } }) => (
              <Select
                options={features}
                isMulti
                value={features.filter((feature) => value?.includes(feature.value))}
                onChange={(selectedOptions) => {
                  onChange(selectedOptions ? selectedOptions.map((option) => option.value) : []);
                }}
              />
            )}
            rules={{ required: true }}
          />
        </div>
        <button className="btn btn-primary btn-validation">Train</button>
        <div>
          <label htmlFor="frequencySlider">
            Refresh the model every {freqRefreshSimpleModel} annotations
          </label>
          <span>5</span>
          <input
            type="range"
            id="frequencySlider"
            min="5"
            max="500"
            onChange={(e) => {
              refreshFreq(Number(e.currentTarget.value));
            }}
            step="1"
          />
          <span>500</span>
        </div>
      </form>
    </div>
  );
};
