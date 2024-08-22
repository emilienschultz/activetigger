import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';

import { useUpdateSimpleModel } from '../core/api';
import { useAppContext } from '../core/context';
import { SimpleModelModel } from '../types';

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
  // element from the context to refresh parameter of the model
  const {
    appContext: { freqRefreshSimpleModel },
    setAppContext,
  } = useAppContext();

  const refreshFreq = (newValue: number) => {
    setAppContext((prev) => ({ ...prev, freqRefreshSimpleModel: newValue }));
  };

  // form management
  const { register, handleSubmit, reset } = useForm<SimpleModelModel>({
    defaultValues: {
      features: [],
      model: '',
      scheme: currentScheme || undefined,
    },
  });

  // state for the model selected
  const [selectedSimpleModel, setSelectedSimpleModel] = useState('');

  // hooks to update
  const { updateSimpleModel } = useUpdateSimpleModel(projectName, currentScheme);

  // action when form validated
  const onSubmit: SubmitHandler<SimpleModelModel> = async (formData) => {
    console.log(formData);
    await updateSimpleModel(formData);
    reset();
    // SET THE VALUE FROM THE STATE
  };

  // define config elements
  const generate_config = (model_name: string): JSX.Element => {
    if (model_name == 'liblinear')
      return (
        <div>
          <label htmlFor="cost">Cost</label>
          <input
            type="number"
            step="1"
            id="cost"
            value={availableSimpleModels.liblinear.cost}
            {...register('params.cost', { valueAsNumber: true })}
          ></input>
        </div>
      );
    if (model_name == 'knn')
      return (
        <div>
          <label htmlFor="n_neighbors">Number of neighbors</label>
          <input
            type="number"
            step="1"
            id="n_neighbors"
            value={availableSimpleModels.knn.n_neighbors}
            {...register('params.n_neighbors', { valueAsNumber: true })}
          ></input>
        </div>
      );
    if (model_name == 'lasso')
      return (
        <div>
          <label htmlFor="c">C</label>
          <input
            type="number"
            step="1"
            id="c"
            value={availableSimpleModels.lasso.c}
            {...register('params.c', { valueAsNumber: true })}
          ></input>
        </div>
      );
    if (model_name == 'multi_naivebayes')
      return (
        <div>
          <label htmlFor="alpha">Alpha</label>
          <input
            type="number"
            id="alpha"
            value={availableSimpleModels.multi_naivebayes.alpha}
            {...register('params.alpha', { valueAsNumber: true })}
          ></input>
          <label htmlFor="fit_prior">Fit prior</label>
          <select
            id="fit_prior"
            value={availableSimpleModels.multi_naivebayes.fit_prior}
            {...register('params.fit_prior')}
          >
            <option>False</option>
            <option>True</option>
          </select>
          <label htmlFor="class_prior">Class prior</label>
          <input
            type="number"
            id="class_prior"
            value={availableSimpleModels.multi_naivebayes.class_prior}
            {...register('params.class_prior')}
          ></input>
        </div>
      );
    if (model_name == 'randomforest')
      return (
        <div>
          <label htmlFor="n_estimators">Number of estimators</label>
          <input
            type="number"
            step="1"
            id="n_estimators"
            value={availableSimpleModels.randomforest.n_estimators}
            {...register('params.n_estimators', { valueAsNumber: true })}
          ></input>
          <label htmlFor="max_features">Max features</label>
          <input
            type="number"
            step="1"
            id="max_features"
            value={availableSimpleModels.randomforest.max_features}
            {...register('params.max_features', { valueAsNumber: true })}
          ></input>
        </div>
      );
    return <div></div>;
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <label htmlFor="model">Select a model</label>
      <select
        id="model"
        {...register('model')}
        onChange={(e) => {
          setSelectedSimpleModel(e.currentTarget.value);
        }}
      >
        {Object.keys(availableSimpleModels).map((e) => (
          <option key={e}>{e}</option>
        ))}{' '}
      </select>
      {generate_config(selectedSimpleModel)}
      <div>
        <label htmlFor="features">Select features</label>
        <select id="features" {...register('features')} multiple>
          {Object.values(availableFeatures).map((e) => (
            <option key={e}>{e}</option>
          ))}{' '}
        </select>
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
  );
};
