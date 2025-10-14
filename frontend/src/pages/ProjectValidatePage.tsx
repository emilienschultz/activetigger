import { FC, useEffect, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import { DisplayScores } from '../components/DisplayScores';
import { DisplayTrainingProcesses } from '../components/DisplayTrainingProcesses';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { useDeleteBertModel, useEvalModel, useModelInformations } from '../core/api';
import { useAppContext } from '../core/context';

/**
 * Component to display the export page
 */
export const ProjectValidatePage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, isComputing, phase },
  } = useAppContext();
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  // available models
  const availableModels = useMemo(() => {
    if (currentScheme && project?.languagemodels?.available?.[currentScheme]) {
      return Object.keys(project.languagemodels.available[currentScheme]);
    }
    return [];
  }, [project, currentScheme]);
  useEffect(() => {
    if (availableModels.length > 0 && !currentModel) {
      setCurrentModel(availableModels[availableModels.length - 1]);
    }
  }, [availableModels, currentModel]);
  const { deleteBertModel } = useDeleteBertModel(projectName || null);
  // hook to api call to launch the test
  const { evalModel } = useEvalModel(
    projectName || null,
    currentScheme || null,
    currentModel || null,
  );
  // get model information from api
  const { model } = useModelInformations(projectName || null, currentModel || null, isComputing);

  return (
    <ProjectPageLayout projectName={projectName} currentAction="validate">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs id="panel" className="mt-3" defaultActiveKey="simple">
              <Tab eventKey="simple" title="Simple">
                {/* <div className="explanations">
                            The simple model is used during tagging, for the active and maxprob
                            models.
                            <a className="problems m-2">
                              <FaTools />
                              <Tooltip anchorSelect=".problems" place="top">
                                Recommended features to train on are embeddings (eg. SBERT) before
                                training a large fine-tuned model, and BERT predictions once you
                                have fine-tuned one.
                              </Tooltip>
                            </a>
                          </div> */}

                {/* <SimpleModelDisplay
                            currentModel={
                              (currentSimpleModel as unknown as Record<string, never>) || undefined
                            }
                          /> */}
              </Tab>
              <Tab eventKey="bert" title="BERT">
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
                        onClick={() => evalModel('valid')}
                        disabled={isComputing}
                      >
                        Compute validation
                      </button>
                      <button
                        className="btn btn-primary m-3"
                        onClick={() => evalModel('test')}
                        disabled={isComputing}
                      >
                        Compute test
                      </button>
                    </div>
                  )}
                  <DisplayTrainingProcesses
                    projectSlug={projectName || null}
                    processes={project?.languagemodels.training}
                    processStatus="testing"
                    displayStopButton={isComputing}
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
                      scores={model.valid_scores as unknown as Record<string, number>}
                      modelName={currentModel || ''}
                      title="Validation scores"
                    />
                  )}

                  {model && (
                    <DisplayScores
                      scores={model.test_scores as unknown as Record<string, number>}
                      modelName={currentModel || ''}
                      title="Test scores"
                    />
                  )}
                </div>
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
