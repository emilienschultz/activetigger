import { FC, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import { DisplayScores } from '../components/DisplayScores';
import { DisplayTrainingProcesses } from '../components/DisplayTrainingProcesses';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { useComputeModelPrediction, useEvalBertModel, useModelInformations } from '../core/api';
import { useAppContext } from '../core/context';
import { ModelDescriptionModel } from '../types';

/**
 * Buttons
 */

interface validateButtonsProps {
  projectSlug: string | null;
  modelName: string | null;
  kind: string | null;
  currentScheme: string | null;
  valid: boolean;
  test: boolean;
  isComputing: boolean;
}

export const ValidateButtons: FC<validateButtonsProps> = ({
  modelName,
  kind,
  valid,
  isComputing,
  test,
  currentScheme,
  projectSlug,
}) => {
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, 16);
  return (
    <div>
      {valid && (
        <button
          className="btn btn-primary m-3"
          onClick={() => computeModelPrediction(modelName || '', 'valid', currentScheme, kind)}
          disabled={isComputing}
        >
          Compute on validation set
        </button>
      )}
      {test && (
        <button
          className="btn btn-primary m-3"
          onClick={() => computeModelPrediction(modelName || '', 'test', currentScheme, kind)}
          disabled={isComputing}
        >
          Compute on test set
        </button>
      )}
    </div>
  );
};

/**
 * Component to display the export page
 */
export const ProjectValidatePage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  // model selected
  const [currentSimpleModelName, setCurrentSimpleModelName] = useState<string | null>(null);
  const [currentBertModelName, setCurrentBertModelName] = useState<string | null>(null);

  // available models
  // const availableBertModels = useMemo(() => {
  //   if (currentScheme && project?.languagemodels?.available?.[currentScheme]) {
  //     return Object.keys(project.languagemodels.available[currentScheme]);
  //   }
  //   return [];
  // }, [project, currentScheme]);

  const availableBertModels = useMemo(
    () =>
      project?.languagemodels.available
        ? project?.languagemodels.available[currentScheme || '']
        : [],
    [project?.languagemodels.available, currentScheme],
  );

  const availableSimpleModels = useMemo(
    () =>
      project?.simplemodel.available
        ? (project?.simplemodel.available[currentScheme || ''] as ModelDescriptionModel[])
        : [],
    [project?.simplemodel.available, currentScheme],
  );

  // hook to api call to launch the test
  const { evalBertModel } = useEvalBertModel(
    projectName || null,
    currentScheme || null,
    currentBertModelName || null,
  );

  // get model information from api
  const { model } = useModelInformations(
    projectName || null,
    currentBertModelName || null,
    isComputing,
  );

  return (
    <ProjectPageLayout projectName={projectName} currentAction="validate">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs id="panel" className="mt-3" defaultActiveKey="simple">
              <Tab eventKey="simple" title="Simple">
                <div>
                  <label htmlFor="selected-model">Existing models</label>
                  <Select
                    options={Object.values(availableSimpleModels || {}).map((e) => ({
                      value: e.name,
                      label: e.name,
                    }))}
                    value={
                      currentSimpleModelName
                        ? { value: currentSimpleModelName, label: currentSimpleModelName }
                        : null
                    }
                    onChange={(selectedOption) => {
                      setCurrentSimpleModelName(selectedOption ? selectedOption.value : null);
                    }}
                    isSearchable
                    className="w-50 mt-1"
                  />
                </div>
                <ValidateButtons
                  modelName={currentSimpleModelName}
                  kind="simple"
                  currentScheme={currentScheme || null}
                  projectSlug={projectName || null}
                  isComputing={isComputing}
                  valid={project?.params.valid || false}
                  test={project?.params.test || false}
                />
              </Tab>
              <Tab eventKey="bert" title="BERT">
                <div>
                  <label htmlFor="selected-model">Existing models</label>
                  <Select
                    options={Object.keys(availableBertModels || {}).map((e) => ({
                      value: e,
                      label: e,
                    }))}
                    value={
                      currentBertModelName
                        ? { value: currentBertModelName, label: currentBertModelName }
                        : null
                    }
                    onChange={(selectedOption) => {
                      setCurrentBertModelName(selectedOption ? selectedOption.value : null);
                    }}
                    isSearchable
                    className="w-50 mt-1"
                  />
                </div>
                <ValidateButtons
                  modelName={currentBertModelName}
                  kind="bert"
                  currentScheme={currentScheme || null}
                  projectSlug={projectName || null}
                  isComputing={isComputing}
                  valid={project?.params.valid || false}
                  test={project?.params.test || false}
                />
                <div>
                  {/* {model && project?.params.valid && (
                    <button
                      className="btn btn-primary m-3"
                      onClick={() =>
                        computeModelPrediction(
                          currentBertModelName || '',
                          'valid',
                          currentScheme || '',
                          'bert',
                        )
                      }
                      disabled={isComputing}
                    >
                      Compute on validation set
                    </button>
                  )}
                  {model && project?.params.test && (
                    <button
                      className="btn btn-primary m-3"
                      onClick={() =>
                        computeModelPrediction(
                          currentBertModelName || '',
                          'test',
                          currentScheme || '',
                          'bert',
                        )
                      }
                      disabled={isComputing}
                    >
                      Compute on test set
                    </button>
                  )} */}

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
                      modelName={currentBertModelName || ''}
                      title="Validation scores"
                    />
                  )}

                  {model && (
                    <DisplayScores
                      scores={model.test_scores as unknown as Record<string, number>}
                      modelName={currentBertModelName || ''}
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
