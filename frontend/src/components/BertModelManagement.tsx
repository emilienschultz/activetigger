import cx from 'classnames';
import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { DisplayTrainingProcesses } from '../components/DisplayTrainingProcesses';
import { ModelCreationForm } from '../components/forms/ModelCreationForm';
import { ModelParametersTab } from '../components/ModelParametersTab';
import { DisplayScores } from './DisplayScores';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { LossChart } from '../components/vizualisation/lossChart';
import { ModelDescriptionModel, ProjectStateModel } from '../types';

import { SubmitHandler, useForm } from 'react-hook-form';
import { useDeleteBertModel, useModelInformations, useRenameBertModel } from '../core/api';
import { useNotifications } from '../core/notifications';
import { MLStatisticsModel } from '../types';

interface renameModel {
  new_name: string;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

interface BertModelManagementProps {
  projectSlug: string | null;
  isComputing: boolean;
  project: ProjectStateModel | null;
  currentScheme: string | null;
  availableBertModels: { [key: string]: ModelDescriptionModel };
}

export const BertModelManagement: FC<BertModelManagementProps> = ({
  projectSlug,
  currentScheme,
  project,
  availableBertModels,
  isComputing,
}) => {
  const { notify } = useNotifications();

  // current model and automatic selection

  const [currentBertModel, setCurrentBertModel] = useState<string | null>(null);
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // get model information from api
  const { model } = useModelInformations(
    projectSlug || null,
    currentBertModel || null,
    'bert',
    isComputing,
  );

  // form to rename
  const { renameBertModel } = useRenameBertModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentBertModel) {
      await renameBertModel(currentBertModel, data.new_name);
      resetRename();
    } else notify({ type: 'error', message: 'New name is void' });
  };

  const loss = model?.loss ? (model?.loss as unknown as LossData) : null;

  const [displayNewBertModel, setDisplayNewBertModel] = useState(false);

  return (
    <div>
      <ModelsPillDisplay
        modelNames={Object.keys(availableBertModels || {}).map((model) => model)}
        currentModelName={currentBertModel}
        setCurrentModelName={setCurrentBertModel}
        deleteModelFunction={deleteBertModel}
      >
        <button
          onClick={() => setDisplayNewBertModel(true)}
          className={cx('model-pill ', isComputing ? 'disabled' : '')}
          id="create-new"
        >
          Create new model
        </button>
      </ModelsPillDisplay>
      {isComputing && (
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={project?.languagemodels.training}
          displayStopButton={isComputing}
        />
      )}
      {currentBertModel && (
        <div>
          {model && (
            <div>
              <details style={{ color: 'gray' }}>
                <summary>
                  <span>Parameters of the model</span>
                </summary>
                <ModelParametersTab params={model.params as Record<string, unknown>} />
              </details>
              <details style={{ color: 'gray' }}>
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
              {isComputing && (
                <DisplayTrainingProcesses
                  projectSlug={projectSlug || null}
                  processes={project?.languagemodels.training}
                  displayStopButton={isComputing}
                />
              )}

              <DisplayScores
                title={'Validation scores from the training data (internal validation)'}
                scores={model.scores.internalvalid_scores as MLStatisticsModel}
                modelName={currentBertModel}
              />

              <div className="mt-2">
                <LossChart loss={loss} />
              </div>
            </div>
          )}
        </div>
      )}

      <Modal
        show={displayNewBertModel}
        id="createmodel-modal"
        size="xl"
        onHide={() => setDisplayNewBertModel(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Train a new BERT model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {/* <div className="explanations">
            The model will be trained on annotated data. A good practice is to have at least 50
            annotated elements. You can exclude elements with specific labels.{' '}
            <a className="problems m-2">
              <FaTools />
              <Tooltip anchorSelect=".problems" place="top">
                If the model doesn't train, the reason can be the limit of available GPU. Please try
                latter. If the problem persists, contact us.
              </Tooltip>
            </a>
          </div> */}
          <ModelCreationForm
            projectSlug={projectSlug || null}
            currentScheme={currentScheme || null}
            project={project || null}
            isComputing={isComputing}
            setStatusDisplay={setDisplayNewBertModel}
          />
        </Modal.Body>
      </Modal>
    </div>
  );
};
