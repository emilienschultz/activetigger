import cx from 'classnames';
import { FC, useEffect, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { MdDriveFileRenameOutline } from 'react-icons/md';
import Select from 'react-select';
import PulseLoader from 'react-spinners/PulseLoader';
import {
  useDeleteQuickModel,
  useGetQuickModel,
  useRenameQuickModel,
  useTrainQuickModel,
} from '../core/api';
import { useNotifications } from '../core/notifications';
import { getRandomName, sortDatesAsStrings } from '../core/utils';
import { MLStatisticsModel, ModelDescriptionModel, QuickModelInModel } from '../types';
import { CreateNewFeature } from './CreateNewFeature';
import { DisplayScores } from './DisplayScores';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { ValidateButtons } from './validateButton';
// TODO: default values + avoid generic parameters

interface Options {
  models?: string[];
}

interface FeaturesOptions {
  fasttext?: Options;
  sbert?: Options;
}

interface QuickModelManagementProps {
  projectName: string | null;
  currentScheme: string | null;
  baseQuickModels: Record<string, Record<string, number>>;
  availableQuickModels: ModelDescriptionModel[];
  availableFeatures: string[];
  availableLabels: string[];
  kindScheme: string;
  currentModel?: Record<string, never>;
  featuresOption: FeaturesOptions;
  columns: string[];
  isComputing: boolean;
}

interface renameModel {
  new_name: string;
}

export default function ModelsTable(
  name: string | null,
  availableQuickModels: ModelDescriptionModel[],
) {
  const model = availableQuickModels.filter((e) => e.name == name)[0];
  if (!model) return null;
  return (
    <>
      <table className="table table-striped table-hover w-50 mt-2">
        <thead>
          <tr>
            <th scope="col">Key</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {model.parameters &&
            Object.entries(model.parameters || {}).map(([key, value], i) => (
              <tr key={i}>
                <td>{key}</td>
                <td>
                  {Array.isArray(value)
                    ? (value as string[]).join(', ') // or use bullets if you prefer
                    : typeof value === 'object' && value !== null
                      ? JSON.stringify(value, null, 2)
                      : String(value)}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </>
  );
}

export const QuickModelManagement: FC<QuickModelManagementProps> = ({
  projectName,
  currentScheme,
  baseQuickModels,
  availableQuickModels,
  availableFeatures,
  availableLabels,
  kindScheme,
  currentModel,
  featuresOption,
  columns,
  isComputing,
}) => {
  const { notify } = useNotifications();

  // hooks to update
  const { trainQuickModel } = useTrainQuickModel(projectName, currentScheme);

  // available features
  const features = availableFeatures.map((e) => ({ value: e, label: e }));

  // current quickmodel
  const [currentQuickModelName, setCurrentQuickModelName] = useState<string | null>(
    availableQuickModels.length > 0 ? availableQuickModels[0].name : null,
  );
  // Modal rename and form to rename
  const [showRename, setShowRename] = useState(false);
  const { renameQuickModel } = useRenameQuickModel(projectName || null);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentQuickModelName) {
      await renameQuickModel(currentQuickModelName, data.new_name);
      resetRename();
      setShowRename(false);
    } else notify({ type: 'error', message: 'New name is void' });
  };

  // get information on the quickmodel
  const { currentModel: currentModelInformations } = useGetQuickModel(
    projectName,
    currentQuickModelName,
    currentQuickModelName,
  );
  const filterFeatures = (features: Feature[]) => {
    const filtered = features.filter((e) => /sbert|fasttext/i.test(e.label));
    const predictFeature = features.find((e) => /predict/i.test(e.label)); // Trouve le premier "predict"
    const sbertFeature = features.find((e) => /sbert/i.test(e.label)); // Trouve le premier "sbert"

    if (sbertFeature) {
      filtered.push(sbertFeature);
    } else if (predictFeature) {
      filtered.push(predictFeature);
    }

    return filtered;
  };

  const predictions = filterFeatures(features);
  const defaultFeatures = predictions.length > 0 ? [predictions[predictions.length - 1]] : [];

  // delete quickmodel
  const { deleteQuickModel } = useDeleteQuickModel(projectName);

  // create form
  const { register, handleSubmit, control, watch, setValue } = useForm<QuickModelInModel>({
    defaultValues: {
      name: getRandomName('QuickModel'),
      model: 'logistic-l1',
      scheme: currentScheme || undefined,
      params: {
        costLogL1: 1,
        costLogL2: 1,
        n_neighbors: 3,
        alpha: 1,
        n_estimators: 500,
        max_features: null,
      },
      dichotomize: kindScheme == 'multilabel' ? availableLabels[0] : undefined,
      features: defaultFeatures.map((e) => e.value),
    },
  });

  // update the values from the current model if it exists
  useEffect(() => {
    if (currentModel?.params) {
      const filteredParams = Object.entries(currentModel.params)
        .filter(([key]) => key !== 'features') // key is the param name
        .reduce(
          (acc, [key, value]) => {
            if (
              typeof value === 'string' ||
              typeof value === 'number' ||
              typeof value === 'boolean'
            ) {
              acc[key] = value;
            }
            return acc;
          },
          {} as Record<string, string | number | boolean>,
        );

      setValue('params', filteredParams as QuickModelInModel['params']);
    }
  }, [currentModel, setValue]);

  // state for the model selected to modify parameters
  const selectedModel = watch('model');

  // action when form validated
  const onSubmit: SubmitHandler<QuickModelInModel> = async (formData) => {
    const watchedFeatures = watch('features');
    if (watchedFeatures.length == 0) {
      notify({ type: 'error', message: 'Please select at least one feature' });
      return;
    }
    await trainQuickModel(formData);
    setDisplayNewModel(false);
  };

  // build default features selected
  type Feature = {
    label: string;
    value: string;
  };

  const [formSelectedFeatures, setFormSelectedFeatures] = useState<string[]>(
    defaultFeatures.map((e) => e.value),
  );

  // state for new feature
  const [displayNewFeature, setDisplayNewFeature] = useState<boolean>(false);
  const [displayNewModel, setDisplayNewModel] = useState<boolean>(false);

  const [showParameters, setShowParameters] = useState(false);

  const selectedFeaturesContainsBERTFeatures = () => {
    return formSelectedFeatures
      .map((feature) => feature?.slice(0, 8) === 'predict_')
      .includes(true);
  };

  const cleanDisplay = (listOfFeatures: string, sep?: string) => {
    if (!sep) {
      sep = ' and ';
    }
    if (listOfFeatures) {
      return listOfFeatures
        .replaceAll('"', '')
        .replaceAll('[', '')
        .replaceAll(']', '')
        .replaceAll(',', sep);
    } else {
      return 'Loading...';
    }
  };

  return (
    <div className="w-100">
      <ModelsPillDisplay
        modelNames={availableQuickModels
          .sort((quickModelA, quickModelB) =>
            sortDatesAsStrings(quickModelA?.time, quickModelB?.time, true),
          )
          .map((quickModel) => quickModel.name)}
        currentModelName={currentQuickModelName}
        setCurrentModelName={setCurrentQuickModelName}
        deleteModelFunction={deleteQuickModel}
      >
        <button
          onClick={() => setDisplayNewModel(true)}
          className={cx('model-pill ', isComputing ? 'disabled' : '')}
          id="create-new"
        >
          <FaPlusCircle size={20} /> Create new model
        </button>
      </ModelsPillDisplay>

      {isComputing && (
        <div className="btn btn-primary mt-3 d-flex align-items-center">
          <PulseLoader color={'white'} /> Computing
        </div>
      )}
      {currentModelInformations && currentQuickModelName && (
        <div>
          <div className="d-flex my-4">
            <button
              className="btn btn-outline-secondary btn-sm me-2 d-flex align-items-center"
              onClick={() => setShowParameters(true)}
            >
              <FaGear size={18} className="me-1" />
              Parameters
            </button>
            <button
              className="btn btn-outline-secondary btn-sm me-2 d-flex align-items-center"
              onClick={() => setShowRename(true)}
            >
              <MdDriveFileRenameOutline size={18} className="me-1" />
              Rename
            </button>
            <ValidateButtons
              projectSlug={projectName}
              modelName={currentQuickModelName}
              kind="quick"
              currentScheme={currentScheme}
              id="compute-prediction"
              buttonLabel="Compute predictions"
            />
          </div>
          <DisplayScores
            title={'Validation scores from the training data (internal validation)'}
            scores={currentModelInformations.statistics_test as MLStatisticsModel}
            projectSlug={projectName}
          />
          {currentModelInformations.statistics_cv10 && (
            <DisplayScores
              title="Cross validation CV10"
              scores={currentModelInformations.statistics_cv10 as unknown as Record<string, number>}
            />
          )}
        </div>
      )}

      <Modal
        show={displayNewModel}
        id="quickmodel-modal"
        onHide={() => setDisplayNewModel(false)}
        centered
        size="xl"
      >
        <Modal.Header closeButton>
          <Modal.Title>Train a new quick model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div>
            <form onSubmit={handleSubmit(onSubmit)}>
              <div>
                <label htmlFor="name">Model name</label>
                <input
                  type="text"
                  id="name"
                  placeholder="Model name"
                  className="form-control"
                  {...register('name')}
                />
              </div>
              <div>
                <div>
                  <label htmlFor="features">Features used to predict (X)</label>
                  <button
                    type="button"
                    className="btn btn-outline-secondary d-flex align-items-center my-1"
                    onClick={() => setDisplayNewFeature(true)}
                  >
                    <FaPlusCircle size={18} className="me-1" /> Add a new feature
                  </button>
                  <Controller
                    name="features"
                    control={control}
                    render={({ field: { onChange, value } }) => (
                      <>
                        {' '}
                        <Select
                          options={features}
                          isMulti
                          value={features.filter((option) => value.includes(option.value))}
                          onChange={(selectedOptions) => {
                            onChange(
                              selectedOptions ? selectedOptions.map((option) => option.value) : [],
                            );
                            setFormSelectedFeatures(
                              selectedOptions ? selectedOptions.map((option) => option.value) : [],
                            );
                          }}
                        />
                      </>
                    )}
                  />
                  {selectedFeaturesContainsBERTFeatures() && (
                    <a className="explanations">
                      ⚠️ Warning: using BERT predictions as features results in strongly
                      upward-biased quality metrics on the train set.
                    </a>
                  )}
                </div>
              </div>
              <details className="custom-details">
                <summary>Advanced parameters</summary>

                <label htmlFor="model">Select a model</label>
                <select id="model" {...register('model')}>
                  {Object.keys(baseQuickModels).map((e) => (
                    <option key={e}>{e}</option>
                  ))}{' '}
                </select>
                {kindScheme == 'multilabel' && (
                  <>
                    <label htmlFor="dichotomize">Dichotomize on the label</label>
                    <select id="dichotomize" {...register('dichotomize')}>
                      {Object.values(availableLabels).map((e) => (
                        <option key={e}>{e}</option>
                      ))}{' '}
                    </select>
                  </>
                )}
                {
                  //generate_config(selectedQuickModel)
                  (selectedModel == 'logistic-l2' && (
                    <div key="logistic-l2">
                      <label htmlFor="costLogL2">Cost</label>
                      <input
                        type="number"
                        step="1"
                        id="logistic-l2"
                        {...register('params.costLogL2', { valueAsNumber: true })}
                      ></input>
                    </div>
                  )) ||
                    (selectedModel == 'knn' && (
                      <div key="knn">
                        <label htmlFor="n_neighbors">Number of neighbors</label>
                        <input
                          type="number"
                          step="1"
                          id="n_neighbors"
                          {...register('params.n_neighbors', { valueAsNumber: true })}
                        ></input>
                      </div>
                    )) ||
                    (selectedModel == 'logistic-l1' && (
                      <div key="logistic-l1">
                        <label htmlFor="costLogL1">Cost</label>
                        <input
                          type="number"
                          step="1"
                          id="logistic-l1"
                          {...register('params.costLogL1', { valueAsNumber: true })}
                        ></input>
                      </div>
                    )) ||
                    (selectedModel == 'multi_naivebayes' && (
                      <div key="multi_naivebayes">
                        <label htmlFor="alpha">Alpha</label>
                        <input
                          type="number"
                          id="alpha"
                          {...register('params.alpha', { valueAsNumber: true })}
                        ></input>
                        <label htmlFor="fit_prior">
                          <input
                            type="checkbox"
                            id="fit_prior"
                            {...register('params.fit_prior')}
                            checked
                          />
                          Fit prior
                        </label>
                      </div>
                    )) ||
                    (selectedModel == 'randomforest' && (
                      <div key="randomforest">
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

                <div className="d-flex align-items-center">
                  <label htmlFor="cv10">
                    <input type="checkbox" id="cv10" {...register('cv10')} />
                    10-fold cross validation
                  </label>
                </div>
              </details>

              <button className="btn btn-primary btn-validation">Train quick model</button>
            </form>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={displayNewFeature}
        id="features-modal"
        size="xl"
        onHide={() => setDisplayNewFeature(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Configure active learning</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <CreateNewFeature
            projectName={projectName || ''}
            featuresOption={featuresOption}
            columns={columns}
          />
        </Modal.Body>
      </Modal>
      <Modal show={showParameters} id="parameters-modal" onHide={() => setShowParameters(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Parameters of {currentQuickModelName}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <table className="table table-striped table-hover w-100 mt-2">
            <tbody>
              <tr>
                <td>Model type</td>
                <td>{currentModelInformations?.model}</td>
              </tr>
              <tr>
                <td>Input features</td>
                <td>
                  {cleanDisplay(
                    JSON.stringify(currentModelInformations?.features) as unknown as string,
                    ', ',
                  )}
                </td>
              </tr>
              {Object.entries(currentModelInformations?.params || {}).map(([key, value], i) => (
                <tr key={i}>
                  <td>{key}</td>
                  <td>
                    {Array.isArray(value)
                      ? (value as string[]).join(', ') // or use bullets if you prefer
                      : typeof value === 'object' && value !== null
                        ? JSON.stringify(value, null, 2)
                        : String(value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Modal.Body>
      </Modal>
      <Modal show={showRename} id="rename-modal" onHide={() => setShowRename(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Rename {currentQuickModelName}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <form onSubmit={handleSubmitRename(onSubmitRename)}>
            <input
              id="new_name"
              className="form-control me-2 mt-2"
              type="text"
              placeholder="New name of the model"
              {...registerRename('new_name')}
            />
            <button className="btn btn-primary me-2 mt-2">Rename</button>
          </form>
        </Modal.Body>
      </Modal>
    </div>
  );
};
