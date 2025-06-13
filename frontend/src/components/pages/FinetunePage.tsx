import { FC, useEffect, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaTools } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import {
  useComputeModelPrediction,
  useDeleteBertModel,
  useModelInformations,
  useRenameBertModel,
  useTestModel,
} from '../../core/api';
import { useAppContext } from '../../core/context';
import { useNotifications } from '../../core/notifications';
import { DisplayScores } from '../DisplayScores';
import { DisplayScoresMenu } from '../DisplayScoresMenu';
import { DisplayTrainingProcesses } from '../DisplayTrainingProcesses';
import { ModelCreationForm } from '../forms/ModelCreationForm';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';
import { ModelParametersTab } from '../ModelParametersTab';
import { ModelPredict } from '../ModelPredict';
import { LossChart } from '../vizualisation/lossChart';

/**
 * Component to manage model training
 */

interface renameModel {
  new_name: string;
}

export const FinetunePage: FC = () => {
  const { projectName: projectSlug } = useParams();

  const { notify } = useNotifications();
  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  const [activeKey, setActiveKey] = useState<string>('models');

  // available models
  const availableModels = useMemo(() => {
    if (currentScheme && project?.languagemodels?.available?.[currentScheme]) {
      return Object.keys(project.languagemodels.available[currentScheme]);
    }
    return [];
  }, [project, currentScheme]);

  const [currentModel, setCurrentModel] = useState<string | null>(null);
  useEffect(() => {
    if (availableModels.length > 0 && !currentModel) {
      setCurrentModel(availableModels[availableModels.length - 1]);
    }
  }, [availableModels, currentModel]);

  const { model } = useModelInformations(projectSlug || null, currentModel || null, isComputing);

  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // compute model preduction
  const [batchSize] = useState<number>(32);
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, batchSize);

  const { testModel } = useTestModel(
    projectSlug || null,
    currentScheme || null,
    currentModel || null,
  );

  // form to rename
  const { renameBertModel } = useRenameBertModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentModel) {
      await renameBertModel(currentModel, data.new_name);
      resetRename();
    } else notify({ type: 'error', message: 'New name is void' });
  };

  // deactivate GPU if not available
  // const { gpu } = useGetServer(project || null);
  // useEffect(() => {
  //   if (!gpu) {
  //     setValue('parameters.gpu', false);
  //   }
  // }, [gpu, setValue]);

  interface LossData {
    epoch: { [key: string]: number };
    val_loss: { [key: string]: number };
    val_eval_loss: { [key: string]: number };
  }

  // loss chart shape data
  const loss = model?.loss ? (model?.loss as unknown as LossData) : null;

  const possibleStatistics = [
    ['Validation (model)', model ? model.valid_scores : null],
    ['Train (model)', model ? model.train_scores : null],
    ['Out of sample', model ? model.outofsample_scores : null],
  ];

  const existingStatistics = Object.fromEntries(
    possibleStatistics.filter(([_, scores]) => scores != null),
  );

  console.log(currentModel);
  console.log(model);

  return (
    <ProjectPageLayout projectName={projectSlug || null} currentAction="finetune">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs
              id="panel"
              className="mt-3"
              activeKey={activeKey}
              onSelect={(k) => setActiveKey(k || 'new')}
            >
              <Tab eventKey="new" title="Create" onSelect={() => setActiveKey('new')}>
                <div className="explanations">
                  The model will be trained on annotated data. A good practice is to have at least
                  50 annotated elements. You can exclude elements with specific labels.{' '}
                  <a className="problems m-2">
                    <FaTools />
                    <Tooltip anchorSelect=".problems" place="top">
                      If the model doesn't train, the reason can be the limit of available GPU.
                      Please try latter. If the problem persists, contact us.
                    </Tooltip>
                  </a>
                </div>
                {isComputing && (
                  <DisplayTrainingProcesses
                    projectSlug={projectSlug || null}
                    processes={project?.languagemodels.training}
                  />
                )}
                <ModelCreationForm
                  projectSlug={projectSlug || null}
                  currentScheme={currentScheme || null}
                  project={project || null}
                  isComputing={isComputing}
                />
              </Tab>
              <Tab
                eventKey="models"
                title="Fine-tuned models"
                onSelect={() => setActiveKey('models')}
              >
                <label htmlFor="selected-model">Existing models</label>
                <div className="d-flex align-items-center">
                  <select
                    id="selected-model"
                    className="form-select"
                    onChange={(e) => setCurrentModel(e.target.value)}
                    value={currentModel || ''}
                  >
                    <option></option>
                    {availableModels.map((e) => (
                      <option key={e}>{e}</option>
                    ))}
                  </select>
                  <button
                    className="btn btn p-0"
                    onClick={() => {
                      if (currentModel) {
                        deleteBertModel(currentModel);
                        setCurrentModel(null);
                      }
                    }}
                  >
                    <MdOutlineDeleteOutline size={30} />
                  </button>
                </div>

                {/* Display the parameters of the selected model */}
                {currentModel && (
                  <div>
                    {model && (
                      <div>
                        <details style={{ color: 'gray' }}>
                          <summary>
                            <span>Parameters of the model</span>
                          </summary>
                          <ModelParametersTab params={model.params as Record<string, unknown>} />
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
                          />
                        )}
                        <button
                          className="btn btn-primary my-2"
                          onClick={() =>
                            computeModelPrediction(currentModel, 'train', currentScheme || '')
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
                              And calculate statistics for the out of sample.<br></br>If the feature
                              already exists, it will be overwritten.
                            </Tooltip>
                          </a>
                        </button>

                        <div className="mt-2">
                          <DisplayScoresMenu scores={existingStatistics} modelName={currentModel} />
                        </div>

                        <div className="mt-2">
                          <LossChart loss={loss} />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </Tab>
              <Tab eventKey="testing" title="Test">
                <div className="explanations">
                  Do not use testset statistics to select the best model, otherwise it’s only a
                  validation set.
                </div>
                {/* Select a model to compute testset predictions */}
                <label htmlFor="selected-model">Existing models</label>
                <div className="d-flex align-items-center">
                  <select
                    id="selected-model"
                    className="form-select"
                    onChange={(e) => setCurrentModel(e.target.value)}
                    value={currentModel || ''}
                  >
                    <option></option>
                    {availableModels.map((e) => (
                      <option key={e}>{e}</option>
                    ))}
                  </select>
                  <button
                    className="btn btn p-0"
                    onClick={() => {
                      if (currentModel) {
                        deleteBertModel(currentModel);
                        setCurrentModel(null);
                      }
                    }}
                  >
                    <MdOutlineDeleteOutline size={30} />
                  </button>
                </div>
                <div>
                  {model && project?.params.test && !isComputing && (
                    <div className="col-12">
                      <button
                        className="btn btn-primary m-3"
                        onClick={() => testModel()}
                        disabled={isComputing}
                      >
                        Compute testset predictions
                      </button>
                    </div>
                  )}
                  <DisplayTrainingProcesses
                    projectSlug={projectSlug || null}
                    processes={project?.languagemodels.training}
                    processStatus="testing"
                  />

                  {model && !project?.params.test && (
                    <div className="col-12">
                      <div className="alert alert-warning m-4">
                        No testset available for this project. Please create one to compute
                        predictions on the project main page
                      </div>
                    </div>
                  )}

                  {model && (
                    <DisplayScores
                      title="Train scores"
                      scores={model.test_scores as unknown as Record<string, number>}
                    />
                  )}
                </div>
              </Tab>
              <Tab eventKey="predict" title="Predict">
                <ModelPredict />
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
