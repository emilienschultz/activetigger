import { FC, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { useParams } from 'react-router-dom';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { useAppContext } from '../core/context';
import { ModelDescriptionModel } from '../types';

import { BertModelManagement } from '../components/BertModelManagement';
import { QuickModelManagement } from '../components/QuickModelManagement';

/**
 * Component to manage model training
 */

export const ProjectModelPage: FC = () => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  const [activeKey, setActiveKey] = useState<string>('quick');

  const baseQuickModels = project?.quickmodel.options ? project?.quickmodel.options : {};

  const availableBertModels = currentScheme && project?.languagemodels.available[currentScheme];
  const availableQuickModels = useMemo(
    () =>
      project?.quickmodel.available
        ? (project?.quickmodel.available as { [key: string]: ModelDescriptionModel[] })
        : {},
    [project?.quickmodel.available],
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

  console.log(availableBertModels);

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="model">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs
              id="panel"
              className="mt-3"
              activeKey={activeKey}
              onSelect={(k) => setActiveKey(k || 'quick')}
            >
              <Tab eventKey="quick" title="Quick">
                <div className="explanations">Train machine learning models based on features</div>

                <QuickModelManagement
                  projectName={projectSlug || null}
                  currentScheme={currentScheme || null}
                  baseQuickModels={
                    baseQuickModels as unknown as Record<string, Record<string, number>>
                  }
                  availableQuickModels={availableQuickModels[currentScheme || ''] || []}
                  availableFeatures={availableFeatures}
                  availableLabels={availableLabels}
                  kindScheme={kindScheme}
                  featuresOption={project?.features.options || {}}
                  columns={project?.params.all_columns || []}
                  isComputing={isComputing}
                />
              </Tab>
              <Tab eventKey="models" title="BERT" onSelect={() => setActiveKey('models')}>
                <div className="explanations">Train BERT models</div>
                <BertModelManagement
                  projectSlug={projectSlug || null}
                  currentScheme={currentScheme || null}
                  availableBertModels={
                    (availableBertModels as unknown as { [key: string]: ModelDescriptionModel }) ||
                    {}
                  }
                  isComputing={isComputing}
                  project={project || null}
                />
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
