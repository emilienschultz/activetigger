import cx from 'classnames';
import { FC, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { GrValidate } from 'react-icons/gr';
import { useParams } from 'react-router-dom';
import { DisplayScoresMenu } from '../components/DisplayScoresMenu';
import { DisplayTrainingProcesses } from '../components/DisplayTrainingProcesses';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
import {
  useComputeModelPrediction,
  useDeleteBertModel,
  useDeleteQuickModel,
  useModelInformations,
} from '../core/api';
import { useAppContext } from '../core/context';
import { MLStatisticsModel, ModelDescriptionModel } from '../types';

/**
 * Buttons
 */

interface validateButtonsProps {
  projectSlug: string | null;
  modelName: string | null;
  kind: string | null;
  currentScheme: string | null;
  setCurrentModel: (val: null | string) => void;
  className?: string;
  id?: string;
}

export const ValidateButtons: FC<validateButtonsProps> = ({
  modelName,
  kind,
  currentScheme,
  projectSlug,
  setCurrentModel,
  className,
  id,
}) => {
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, 16);
  return (
    <button
      className={cx(className ? className : 'btn btn-primary m-4')}
      onClick={() => {
        computeModelPrediction(modelName || '', 'annotable', currentScheme, kind);
        setCurrentModel(null);
      }}
      id={id}
    >
      <GrValidate size={20} /> Compute statistics on annotations
    </button>
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
  const [currentQuickModelName, setCurrentQuickModelName] = useState<string | null>(null);
  const [currentBertModelName, setCurrentBertModelName] = useState<string | null>(null);
  // delete quickmodel
  const { deleteQuickModel } = useDeleteQuickModel(projectName as string);
  const { deleteBertModel } = useDeleteBertModel(projectName as string);

  const { model: bertModelInformations } = useModelInformations(
    projectName || null,
    currentBertModelName || null,
    'bert',
    isComputing,
  );

  // get model information from api
  const { model: quickModelInformations } = useModelInformations(
    projectName || null,
    currentQuickModelName || null,
    'quick',
    isComputing,
  );

  const availableBertModels = useMemo(
    () =>
      project?.languagemodels.available
        ? project?.languagemodels.available[currentScheme || '']
        : [],
    [project?.languagemodels.available, currentScheme],
  );

  const availableQuickModels = useMemo(
    () =>
      project?.quickmodel.available
        ? (project?.quickmodel.available[currentScheme || ''] as ModelDescriptionModel[])
        : [],
    [project?.quickmodel.available, currentScheme],
  );

  return (
    <ProjectPageLayout projectName={projectName} currentAction="validate">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs id="panel" className="mt-3" defaultActiveKey="quick">
              <Tab eventKey="quick" title="Quick">
                <div className="explanations">
                  Compute statistics on annotations for machine learning models
                </div>
                {availableQuickModels && (
                  <ModelsPillDisplay
                    modelNames={(availableQuickModels || {})?.map((model) => model.name)}
                    currentModelName={currentQuickModelName}
                    setCurrentModelName={setCurrentQuickModelName}
                    deleteModelFunction={deleteQuickModel}
                  ></ModelsPillDisplay>
                )}

                {quickModelInformations && (
                  <>
                    <ValidateButtons
                      modelName={currentQuickModelName}
                      kind="quick"
                      currentScheme={currentScheme || null}
                      projectSlug={projectName || null}
                      setCurrentModel={setCurrentQuickModelName}
                      id="compute-validate"
                    />

                    <DisplayScoresMenu
                      scores={
                        quickModelInformations.scores as unknown as Record<
                          string,
                          MLStatisticsModel
                        >
                      }
                      modelName={currentQuickModelName || ''}
                      skip={['internalvalid_scores']}
                    />
                  </>
                )}
              </Tab>
              <Tab eventKey="bert" title="BERT">
                <div className="explanations">
                  Compute statistics on annotations for BERT models
                </div>
                {availableQuickModels && (
                  <ModelsPillDisplay
                    modelNames={Object.keys(availableBertModels || {})?.map((model) => model)}
                    currentModelName={currentBertModelName}
                    setCurrentModelName={setCurrentBertModelName}
                    deleteModelFunction={deleteBertModel}
                  ></ModelsPillDisplay>
                )}
                <div>
                  {/* AM: Necessary ? Confused... */}
                  <DisplayTrainingProcesses
                    projectSlug={projectName || null}
                    processes={project?.languagemodels.training}
                    processStatus="testing"
                    displayStopButton={isComputing}
                  />

                  {bertModelInformations && !project?.params.test && (
                    <div className="col-12">
                      <div className="alert alert-warning m-4">
                        No testset available for this project. Please create one to compute
                        predictions on the project main page
                      </div>
                    </div>
                  )}

                  {bertModelInformations && (
                    <>
                      <ValidateButtons
                        modelName={currentBertModelName}
                        kind="bert"
                        currentScheme={currentScheme || null}
                        projectSlug={projectName || null}
                        setCurrentModel={setCurrentBertModelName}
                        id="compute-validate"
                      />
                      <DisplayScoresMenu
                        scores={
                          bertModelInformations.scores as unknown as Record<
                            string,
                            MLStatisticsModel
                          >
                        }
                        modelName={currentQuickModelName || ''}
                        skip={['internalvalid_scores']}
                      />
                    </>
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
