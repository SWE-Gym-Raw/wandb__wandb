# Generated by ariadne-codegen
# Source: wandb/sdk/automations/queries.graphql

__all__ = [
    "CREATE_TRIGGER_GQL",
    "DELETE_TRIGGER_GQL",
    "SLACK_INTEGRATIONS_FOR_TEAM_GQL",
    "SLACK_INTEGRATIONS_FOR_USER_GQL",
    "TRIGGERS_IN_USER_ORGS_GQL",
]

TRIGGERS_IN_USER_ORGS_GQL = """
query TriggersInUserOrgs($entityName: String) {
  viewer(entityName: $entityName) {
    organizations {
      orgEntity {
        projects {
          edges {
            node {
              triggers {
                ...Trigger
              }
            }
          }
        }
      }
      teams {
        projects {
          edges {
            node {
              triggers {
                ...Trigger
              }
            }
          }
        }
      }
    }
  }
}

fragment ArtifactPortfolioScope on ArtifactPortfolio {
  __typename
  id
  name
}

fragment ArtifactSequenceScope on ArtifactSequence {
  __typename
  id
  name
}

fragment FilterEventTriggeringCondition on FilterEventTriggeringCondition {
  __typename
  eventType
  filter
}

fragment NotificationAction on NotificationTriggeredAction {
  __typename
  title
  message
  severity
  integration {
    ...SlackIntegration
  }
}

fragment ProjectScope on Project {
  __typename
  id
  name
}

fragment QueueJobAction on QueueJobTriggeredAction {
  __typename
  template
  queue {
    ...RunQueue
  }
}

fragment RunQueue on RunQueue {
  id
  name
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}

fragment Trigger on Trigger {
  __typename
  id
  createdAt
  createdBy {
    ...UserInfo
  }
  updatedAt
  name
  description
  enabled
  scope {
    ...ProjectScope
    ...ArtifactPortfolioScope
    ...ArtifactSequenceScope
  }
  triggeringCondition {
    ...FilterEventTriggeringCondition
  }
  triggeredAction {
    ...QueueJobAction
    ...NotificationAction
    ...WebhookAction
  }
}

fragment UserInfo on User {
  id
  username
}

fragment WebhookAction on GenericWebhookTriggeredAction {
  __typename
  requestPayload
  integration {
    ...WebhookIntegration
  }
}

fragment WebhookIntegration on GenericWebhookIntegration {
  __typename
  id
  name
  urlEndpoint
  secretRef
  accessTokenRef
  createdAt
}
"""

CREATE_TRIGGER_GQL = """
mutation CreateTrigger($params: CreateFilterTriggerInput!) {
  createFilterTrigger(input: $params) {
    ...CreateFilterTriggerResult
  }
}

fragment ArtifactPortfolioScope on ArtifactPortfolio {
  __typename
  id
  name
}

fragment ArtifactSequenceScope on ArtifactSequence {
  __typename
  id
  name
}

fragment CreateFilterTriggerResult on CreateFilterTriggerPayload {
  __typename
  trigger {
    ...Trigger
  }
  clientMutationId
}

fragment FilterEventTriggeringCondition on FilterEventTriggeringCondition {
  __typename
  eventType
  filter
}

fragment NotificationAction on NotificationTriggeredAction {
  __typename
  title
  message
  severity
  integration {
    ...SlackIntegration
  }
}

fragment ProjectScope on Project {
  __typename
  id
  name
}

fragment QueueJobAction on QueueJobTriggeredAction {
  __typename
  template
  queue {
    ...RunQueue
  }
}

fragment RunQueue on RunQueue {
  id
  name
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}

fragment Trigger on Trigger {
  __typename
  id
  createdAt
  createdBy {
    ...UserInfo
  }
  updatedAt
  name
  description
  enabled
  scope {
    ...ProjectScope
    ...ArtifactPortfolioScope
    ...ArtifactSequenceScope
  }
  triggeringCondition {
    ...FilterEventTriggeringCondition
  }
  triggeredAction {
    ...QueueJobAction
    ...NotificationAction
    ...WebhookAction
  }
}

fragment UserInfo on User {
  id
  username
}

fragment WebhookAction on GenericWebhookTriggeredAction {
  __typename
  requestPayload
  integration {
    ...WebhookIntegration
  }
}

fragment WebhookIntegration on GenericWebhookIntegration {
  __typename
  id
  name
  urlEndpoint
  secretRef
  accessTokenRef
  createdAt
}
"""

DELETE_TRIGGER_GQL = """
mutation DeleteTrigger($id: ID!) {
  deleteTrigger(input: {triggerID: $id}) {
    ...DeleteTriggerResult
  }
}

fragment DeleteTriggerResult on DeleteTriggerPayload {
  __typename
  success
  clientMutationId
}
"""

SLACK_INTEGRATIONS_FOR_USER_GQL = """
query SlackIntegrationsForUser {
  viewer {
    userEntity {
      integrations {
        edges {
          node {
            __typename
            ...SlackIntegration
          }
        }
      }
    }
  }
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}
"""

SLACK_INTEGRATIONS_FOR_TEAM_GQL = """
query SlackIntegrationsForTeam($entityName: String!) {
  entity(name: $entityName) {
    integrations {
      edges {
        node {
          __typename
          ...SlackIntegration
        }
      }
    }
  }
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}
"""
