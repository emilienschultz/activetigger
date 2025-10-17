import { FC, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaTools } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { DisplayTrainingProcesses } from '../components/DisplayTrainingProcesses';
import { ModelCreationForm } from '../components/forms/ModelCreationForm';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelParametersTab } from '../components/ModelParametersTab';
import { ModelPredict } from '../components/ModelPredict';
import { LossChart } from '../components/vizualisation/lossChart';
import {
  useComputeBertModelPrediction,
  useDeleteBertModel,
  useModelInformations,
  useRenameBertModel,
} from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { ModelDescriptionModel } from '../types';

import { DisplayScores } from '../components/DisplayScores';
import { SimpleModelManagement } from '../components/SimpleModelManagement';
import { MLStatisticsModel } from '../types';

/**
 * Component to manage model training
 */

interface renameModel {
  new_name: string;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

export const ProjectModelPage: FC = () => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, isComputing, phase },
  } = useAppContext();
  const { notify } = useNotifications();

  const [activeKey, setActiveKey] = useState<string>('simple');

  // available models
  const availableBertModels = useMemo(() => {
    if (currentScheme && project?.languagemodels?.available?.[currentScheme]) {
      return Object.keys(project.languagemodels.available[currentScheme]);
    }
    return [];
  }, [project, currentScheme]);
  const baseSimpleModels = project?.simplemodel.options ? project?.simplemodel.options : {};

  const availableSimpleModels = useMemo(
    () =>
      project?.simplemodel.available
        ? (project?.simplemodel.available as { [key: string]: ModelDescriptionModel[] })
        : {},
    [project?.simplemodel.available],
  );
  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass',
  );

  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // compute model prediction
  const [batchSize, setBatchSize] = useState<number>(32);
  const { computeBertModelPrediction } = useComputeBertModelPrediction(
    projectSlug || null,
    batchSize,
  );

  // form to rename
  const { renameBertModel } = useRenameBertModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  // current model and automatic selection
  const [currentBertModel, setCurrentBertModel] = useState<string | null>(null);
  const [currentSimpleModel, setCurrentSimpleModel] = useState<string | null>(null);

  // get model information from api
  const { model } = useModelInformations(
    projectSlug || null,
    currentBertModel || null,
    isComputing,
  );

  // loss chart shape data
  const loss = model?.loss ? (model?.loss as unknown as LossData) : null;

  // display statistics options
  const possibleStatistics = [
    ['Validation (model)', model ? model.valid_scores : null],
    ['Train (model)', model ? model.train_scores : null],
    ['Out of sample', model ? model.outofsample_scores : null],
  ];
  const existingStatistics = Object.fromEntries(
    possibleStatistics.filter(([_, scores]) => scores != null),
  );

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentBertModel) {
      await renameBertModel(currentBertModel, data.new_name);
      resetRename();
    } else notify({ type: 'error', message: 'New name is void' });
  };

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="model">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs
              id="panel"
              className="mt-3"
              activeKey={activeKey}
              onSelect={(k) => setActiveKey(k || 'simple')}
            >
              <Tab eventKey="simple" title="Simple">
                <div className="container-fluid">
                  <div className="row">
                    <SimpleModelManagement
                      projectName={projectSlug || null}
                      currentScheme={currentScheme || null}
                      baseSimpleModels={
                        baseSimpleModels as unknown as Record<string, Record<string, number>>
                      }
                      availableSimpleModels={availableSimpleModels[currentScheme || ''] || []}
                      availableFeatures={availableFeatures}
                      availableLabels={availableLabels}
                      kindScheme={kindScheme}
                      currentModel={
                        (currentSimpleModel as unknown as Record<string, never>) || undefined
                      }
                    />
                  </div>
                </div>
              </Tab>
              <Tab eventKey="models" title="BERT" onSelect={() => setActiveKey('models')}>
                <Tabs id="bert" className="mt-1" defaultActiveKey="existing">
                  <Tab eventKey="existing" title="Existing">
                    <label htmlFor="selected-model">Existing models</label>
                    <div className="d-flex align-items-center">
                      <select
                        id="selected-model"
                        className="form-select"
                        onChange={(e) => setCurrentBertModel(e.target.value)}
                        value={currentBertModel || ''}
                      >
                        <option></option>
                        {availableBertModels.map((e) => (
                          <option key={e}>{e}</option>
                        ))}
                      </select>
                      <button
                        className="btn btn p-0"
                        onClick={() => {
                          if (currentBertModel) {
                            deleteBertModel(currentBertModel);
                            setCurrentBertModel(null);
                          }
                        }}
                      >
                        <MdOutlineDeleteOutline size={30} />
                      </button>
                    </div>

                    {currentBertModel && (
                      <div>
                        {model && (
                          <div>
                            <details style={{ color: 'gray' }}>
                              <summary>
                                <span>Parameters of the model</span>
                              </summary>
                              <div className="d-flex align-items-center">
                                <label>Batch size</label>
                                <a className="batch">
                                  <HiOutlineQuestionMarkCircle />
                                </a>
                                <Tooltip anchorSelect=".batch" place="top">
                                  Batch used for predict. Keep it small (16 or 32) for small GPU.
                                </Tooltip>
                                <input
                                  type="number"
                                  step="1"
                                  className="m-2"
                                  style={{ width: '50px' }}
                                  value={batchSize}
                                  onChange={(e) => setBatchSize(Number(e.target.value))}
                                />
                              </div>
                              <ModelParametersTab
                                params={model.params as Record<string, unknown>}
                              />
                              <details className="m-2">
                                <summary>Rename</summary>
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
                              </details>
                            </details>
                            {isComputing && (
                              <DisplayTrainingProcesses
                                projectSlug={projectSlug || null}
                                processes={project?.languagemodels.training}
                                displayStopButton={isComputing}
                              />
                            )}
                            {/* <button
                              className="btn btn-primary my-2"
                              onClick={() =>
                                computeBertModelPrediction(
                                  currentBertModel,
                                  'train',
                                  currentScheme || '',
                                )
                              }
                              disabled={isComputing}
                            >
                              Transform to feature
                              <a className="toFeature">
                                <HiOutlineQuestionMarkCircle
                                  style={{ color: 'white' }}
                                  className="mx-2"
                                />
                                <Tooltip anchorSelect=".toFeature" place="top">
                                  And calculate statistics for the out of sample.<br></br>If the
                                  feature already exists, it will be overwritten.
                                </Tooltip>
                              </a>
                            </button> */}

                            <div className="mt-2">
                              <DisplayScores
                                title={'Internal validation'}
                                scores={model.internalvalid_scores as MLStatisticsModel}
                                modelName={currentBertModel}
                              />
                            </div>

                            <div className="mt-2">
                              <LossChart loss={loss} />
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </Tab>
                  <Tab eventKey="new" title="New">
                    <div className="explanations">
                      The model will be trained on annotated data. A good practice is to have at
                      least 50 annotated elements. You can exclude elements with specific labels.{' '}
                      <a className="problems m-2">
                        <FaTools />
                        <Tooltip anchorSelect=".problems" place="top">
                          If the model doesn't train, the reason can be the limit of available GPU.
                          Please try latter. If the problem persists, contact us.
                        </Tooltip>
                      </a>
                    </div>
                    <DisplayTrainingProcesses
                      projectSlug={projectSlug || null}
                      processes={project?.languagemodels.training}
                      displayStopButton={isComputing}
                    />

                    <ModelCreationForm
                      projectSlug={projectSlug || null}
                      currentScheme={currentScheme || null}
                      project={project || null}
                      isComputing={isComputing}
                    />
                  </Tab>
                  <Tab eventKey="predict" title="Predict">
                    <ModelPredict />
                  </Tab>
                </Tabs>
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
